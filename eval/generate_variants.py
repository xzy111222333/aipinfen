import argparse
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from scoring.engine import call_llm_json, clean_reference_body, count_exam_grids
from scoring.rubric_cache import save_json

from .prepare_dataset import EVAL, load_bundle, read_json
from .variant_prompts import GENERATOR, VARIANTS, VERIFIER, VERSION


def applicable_codes(question, stem):
    if question["genre"] == "大作文":
        return [code for code in VARIANTS if code.startswith("E")]
    codes = [code for code in VARIANTS if not code.startswith("E")]
    if question["genre"] not in {"归纳概括", "公文", "短评"}:
        codes.remove("D4")
    if re.search(r"(?:按|按照)(?:时间|先后|时间线|先后顺序)", stem):
        codes.remove("V3")
    if question["limit"] is None:
        codes.remove("D8")
    return codes


def _generate(context, codes, attempt):
    return call_llm_json(GENERATOR, {**context, "变体": {code: VARIANTS[code][0] for code in codes}},
                         temperature=0.45 + 0.10 * attempt, max_tokens=12000)


def _verify(context, question, note, text, code):
    if not text.strip():
        return False, "正文为空", None
    if text.strip() == context["底稿"].strip():
        return False, "改稿与底稿完全相同", None
    if code == "D8" and (not text.startswith(context["底稿"]) or
                         count_exam_grids(text) < math.ceil(question["limit"] * 1.2)):
        return False, "未完整保留底稿前缀或超字数不足 20%", None
    check = call_llm_json(VERIFIER, {**context, "唯一改写要求": VARIANTS[code][0], "改稿": text},
                          temperature=0.1, max_tokens=900)
    verified = check.get("verified") is True and isinstance(check.get("note"), str) and bool(check["note"].strip())
    core_count = check.get("core_count")
    if code == "D1" and (type(core_count) is not int or core_count < 1):
        return False, "单点删除缺少独立核验的核心要点数", None
    return verified, check.get("note") or note or "核验没有返回具体依据", core_count


def build_variants(question, base):
    bundle = load_bundle(question)
    context = {"题干与要求": bundle["题干与要求"], "材料": bundle["材料段落"],
               "字数上限": question["limit"], "底稿": clean_reference_body(base["text"])}
    folder = EVAL / "variants" / question["id"]
    folder.mkdir(parents=True, exist_ok=True)
    codes = applicable_codes(question, context["题干与要求"])
    pending, existing = [], {}
    for code in codes:
        path = folder / f"{base['id']}-{code}.json"
        if path.is_file():
            existing[code] = read_json(path)
        if code not in existing or (not existing[code]["verified"] and existing[code]["attempts"] < 3):
            pending.append(code)
    initial = _generate(context, [code for code in pending if code not in existing], 0) if any(
        code not in existing for code in pending) else {}
    passed = sum(item["verified"] for item in existing.values())
    for code in pending:
        identifier = f"{base['id']}-{code}"
        path = folder / f"{identifier}.json"
        first_attempt = existing.get(code, {}).get("attempts", 0)
        history = existing.get(code, {}).get("verify_history", [])
        expect = dict(VARIANTS[code][1])
        for attempt in range(first_attempt, 3):
            proposal = initial.get(code, {}) if attempt == 0 else _generate(context, [code], attempt).get(code, {})
            text = proposal.get("text", "") if isinstance(proposal, dict) else ""
            note = proposal.get("note", "") if isinstance(proposal, dict) else ""
            if not isinstance(text, str):
                text = ""
            verified, verification, cores = _verify(context, question, note, text, code)
            if verified and code == "D1":
                expect["min_drop_ratio"] = expect["removed_point_fraction"] / cores
                expect["core_count_proxy"] = cores
            history.append(verification)
            save_json(path, {
                "variant_id": identifier, "question_id": question["id"], "base_id": base["id"],
                "type": code, "text": text if verified else "", "expect": expect,
                "verified": verified, "verify_note": verification, "verify_history": history,
                "attempts": attempt + 1, "generator_prompt_version": VERSION,
            })
            if verified:
                passed += 1
                break
    return passed, len(codes)


def main():
    parser = argparse.ArgumentParser(description="生成并独立核验评测对照答卷，可接续已保存结果")
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--limit", type=int, help="仅处理此集合前若干题，用于试跑")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    questions = [q for q in read_json(EVAL / "questions.json") if q["split"] == args.split][:args.limit]
    by_id = {q["id"]: q for q in questions}
    bases = [b for b in read_json(EVAL / "bases.json") if b["question_id"] in by_id]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(build_variants, by_id[b["question_id"]], b): b["id"] for b in bases}
        for job in as_completed(jobs):
            try:
                passed, total = job.result()
                print(f"{jobs[job]} 核验 {passed}/{total}", flush=True)
            except (OSError, RuntimeError, ValueError) as exc:
                failures.append(f"{jobs[job]} {type(exc).__name__}: {exc}")
                print(failures[-1], flush=True)
    if failures:
        parser.exit(1, "有调用失败，已核验的变体可接续使用。\n")


if __name__ == "__main__":
    main()
