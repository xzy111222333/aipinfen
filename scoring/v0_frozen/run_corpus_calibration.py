import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .paper_pipeline import prefab_full_paper
from .rubric_cache import DEFAULT_CACHE_DIR, save_json
from .run_batch_calibration import evaluate_recalled_paper


def _paper_identity(text):
    year = re.search(r"(20\d{2})年", text)
    track = re.search(r"副省|地市|行政执法", text)
    return (year[1], track[0]) if year and track else None


def _group_samples(paper_root, files):
    papers = {}
    for path in paper_root.iterdir():
        if path.is_dir() and (identity := _paper_identity(path.name)):
            if identity in papers:
                raise ValueError(f"试卷身份重复：{path.name} 与 {papers[identity].name}")
            papers[identity] = path
    groups, failures = {}, []
    for path in sorted(set(files)):
        header = path.read_text(encoding="utf-8-sig").split("## 第", 1)[0]
        match = re.search(r"(?m)^- 试卷[：:]\s*(.+)", header)
        paper = papers.get(_paper_identity(match[1])) if match else None
        if paper is None:
            failures.append({"答卷": str(path), "错误": "未找到与卷别相符的试卷"})
        else:
            groups.setdefault(paper, []).append(path)
    return groups, failures


def _score_group(paper, paths, ref_limit, cache_dir):
    try:
        prepared = prefab_full_paper(paper, ref_limit=ref_limit, cache_dir=cache_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        return [], [{"答卷": str(path), "错误": f"预制失败：{exc}"} for path in paths]

    results, failures = [], []
    for path in paths:
        try:
            row = evaluate_recalled_paper(prepared, path)
            results.append({"试卷": paper.name, "来源包": path.parent.parent.name, **row})
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append({"答卷": str(path), "错误": f"批改失败：{exc}"})
    return results, failures


def main():
    parser = argparse.ArgumentParser(description="批量校验多个卷别的申论回忆卷")
    parser.add_argument("papers", type=Path, help="试卷卷别目录的共同父目录")
    parser.add_argument("corpus", type=Path, nargs="+", help="答卷文件，或直接存放答卷 Markdown 的目录")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--ref-limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.ref_limit < 0 or args.workers < 1:
        parser.error("机构答案上限不能为负，并发数至少为 1")
    files = [file for path in args.corpus for file in (path.glob("*.md") if path.is_dir() else [path])]
    if not files:
        parser.error("没有找到答卷 Markdown")
    groups, failures = _group_samples(args.papers, files)
    results = []
    report = {"样本数": len(set(files)), "机构答案上限": args.ref_limit, "结果": results, "失败": failures}
    save_json(args.output, report)
    print(f"匹配到 {len(groups)} 套试卷，待批改 {sum(map(len, groups.values()))} 份答卷", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {
            pool.submit(_score_group, paper, paths, args.ref_limit, args.cache_dir): paper
            for paper, paths in groups.items()
        }
        for job in as_completed(jobs):
            scored, failed = job.result()
            results.extend(scored)
            failures.extend(failed)
            save_json(args.output, report)
            print(f"{jobs[job].name} 完成 {len(scored)}，失败 {len(failed)}，累计完成 {len(results)}", flush=True)
    if failures:
        parser.exit(1, f"有 {len(failures)} 份未完成，原因见报告。\n")


if __name__ == "__main__":
    main()
