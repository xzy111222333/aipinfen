"""冻结细则后，各份对照答卷独立送评。可重复运行补全中断的回放。"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rubric_workshop.blind_batch import OUT, evaluate, load, save


CASES = {
    '2025地市-1': ('正向条件回测/2025地市-1', '答卷1 答卷2 答卷3 答卷4 删储碳'),
    '2023地市-1': ('正向条件回测/2023地市-1', '原稿 等义转述 删融资 套话稿'),
    '2026地市-3': ('正向条件回测/2026地市-3', '答卷1 答卷2 答卷3 答卷4 答卷5 删培训形式 删报名指引'),
    '2023地市-3': ('光伏分档盲评/默认', '小张原稿 中档对照 套话对照 替代维权措施 反向建议稿'),
    '2023地市-4': ('短评格式对照', '四海原稿 四海去序号 袁东原稿 独立短评'),
    '2024副省-2': ('正向条件回测/2024副省-2', '盲写原稿 原段倒序'),
}


def replay(directory, key, name, answer):
    record = load(directory/key/'细则.json')
    path = directory/key/'试评'/(name+'.json')
    if path.exists():
        if load(path)['answer'] != answer:
            raise ValueError(f'答卷已变更 {path}')
        return
    marks = evaluate(record['source'], record['rubric'], answer)
    save(path, dict(answer=answer, marks=marks, total=sum(m['score'] for m in marks)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('keys', nargs='*')
    args = parser.parse_args()
    errors = []
    with ThreadPoolExecutor(6) as pool:
        jobs = {}
        for key in args.keys or CASES:
            folder, names = CASES[key]
            answers = {n: load(OUT/folder/(n+'.json'))['answer'] for n in names.split()}
            if key == '2023地市-4':
                answers['删材料论据'] = '\n'.join(answers['独立短评'].splitlines()[:2])
            for name, answer in answers.items():
                jobs[pool.submit(replay, args.directory, key, name, answer)] = (key, name)
        for f in as_completed(jobs):
            try:
                f.result()
            except Exception as error:
                errors.append((jobs[f], str(error)))
    if errors:
        raise RuntimeError(errors)


if __name__ == '__main__':
    main()
