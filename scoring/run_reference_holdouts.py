"""按高分题库档案隔离机构稿并复评，结果按题落盘。"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

from .class_a_pipeline import call_preparation_responses
from .engine import RELAY_MODEL, load_question_bundle, parse_hard_constraints
from .reference_workflow import build, clean_body_and_votes, digest, save, strip_editorial_markers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples', type=Path)
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('questions', nargs='+', help='题库键名与版本，如 2025地市-1@1')
    parser.add_argument('--preparation-model', required=True)
    parser.add_argument('--high-ratio', type=float, default=0.8)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if not (0 < args.high_ratio <= 1) or args.workers < 1:
        parser.error('高分比例须在 (0, 1] 内，并发数至少为 1')
    samples = {(row['key'], str(row['version'])): row for row in json.loads(args.samples.read_text(encoding='utf-8'))}
    jobs = {}
    for selection in args.questions:
        key, version = selection.rsplit('@', 1)
        if key in jobs or (key, version) not in samples:
            parser.error(f'题目重复或版本不存在：{selection}')
        row = samples[key, version]
        question = args.archive.joinpath(*row['source'].replace('\\', '/').split('/')[1:]).resolve(strict=True)
        maximum = parse_hard_constraints(load_question_bundle(question)['题干与要求'])['满分']
        if maximum != row['max_score']:
            raise ValueError(f'{key}满分与高分库不符')
        held = {'原编号': -1, '来源': row['author'], **clean_body_and_votes(strip_editorial_markers(row['answer']))}
        jobs[key] = (question, held, row, round(maximum * args.high_ratio, 4))

    manifest = {
        '制标与成稿': args.preparation_model, '匿名阅卷': RELAY_MODEL, '高分比例': args.high_ratio,
        '题目': {key: {'源文件': str(q), '留出作者': held['来源'], '留出版本': row['version'],
                      '题库原稿指纹': row['sha256'], '阅卷输入指纹': digest(held['正文']), '高分线': line}
                 for key, (q, held, row, line) in jobs.items()},
    }
    manifest_path = args.output / '验收锁定.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('当前目录已有不同的验收清单')
    save(manifest_path, manifest)
    prepare = partial(call_preparation_responses, args.preparation_model)

    def run(key):
        question, held, _, line = jobs[key]
        result = build(question, None, None, args.output / key, external_holdout=[held],
                       preparation_model=args.preparation_model, preparation_call=prepare)
        marks = result['盲评结果']
        if marks['完美标杆']['细则指纹'] != marks[held['来源']]['细则指纹']:
            raise ValueError(f'{key}的两份评分细则指纹不一致')
        save(args.output / key / '全流程结果.json', result)
        info = {'标杆分': marks['完美标杆']['总分'], '标杆格数': result['标杆格数'],
                '留出分': marks[held['来源']]['总分'], '高分线': line, '达标': marks[held['来源']]['总分'] >= line}
        print(key, info, flush=True)
        return info

    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run, key): key for key in jobs}
        for task in as_completed(pending):
            key = pending[task]
            try:
                results[key] = task.result()
            except Exception as error:
                results[key] = {'运行错误': f'{type(error).__name__}: {error}'}
                print(key, results[key], flush=True)
            save(args.output / '验收结果.json', results)
    if any('运行错误' in row for row in results.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
