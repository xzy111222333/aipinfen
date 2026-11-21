"""只凭题源独立作答，按同一冻结表盲评完整样稿、参照稿和既有对照稿。"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

from rubric_workshop.blind_batch import (
    OUT, call_llm_json, catalog, count_pure_exam_chars, evaluate, load,
    measure_grid_sheet, save, core_rubric,
)
from rubric_workshop.joint_replay import cases_for


def write_answer(source, style):
    cap = int(re.search(r"不超过\s*(\d+)\s*字", source['question'])[1])
    payload = dict(source, 建议篇幅=f"先按{int(cap*0.83)}个含标点的格数写完，法定上限{cap}格。")
    for _ in range(3):
        answer = call_llm_json('作为考生独立完整作答。' + style + '按题面限字和文种留出排版空间。只返回 {"answer":"答卷正文"}。', payload)['answer']
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError('样稿为空')
        size = count_pure_exam_chars(answer)
        sheet = measure_grid_sheet(answer, cap, buffer_rows=0)
        blanks = sum(not s.strip() for s in answer.strip().splitlines())
        if size <= cap and (source['type'] == 1 or sheet['方格纸实占行数'] + blanks <= sheet['标准上限行数'] and not sheet['截断溢出正文']):
            return answer
        payload['上次答卷'] = answer
        payload['修订'] = f'上次计{size}格；请至少再删{max(12,size-int(cap*0.83))}字并减少分段。'
    raise ValueError('独立样稿超过题目字限或估算排格')


def compare_answers(original, variant, equivalence, allow_equal=False):
    if [m['name'] for m in original['marks']] != [m['name'] for m in variant['marks']]:
        raise ValueError('对照答卷使用的细则不同')
    changes = []
    retained = []
    for left, right in zip(original['marks'], variant['marks']):
        for before, after in zip(left['decisions'], right['decisions'], strict=True):
            if before['condition'] != after['condition']:
                raise ValueError('对照答卷的子分条件不同')
            if before['score'] != after['score']:
                changes.append(dict(item=left['name'],condition=before['condition'],before=before['score'],after=after['score']))
                quotes=before.get('citations') or [before.get('quote','')]
                if before['score'] > after['score'] and all(q and q in variant['answer'] for q in quotes):
                    retained.append(dict(item=left['name'], condition=before['condition'], evidence=quotes))
    passed = not changes if equivalence else (variant['score'] < original['score'] and all(d['after'] < d['before'] for d in changes))
    if allow_equal and not changes:
        passed = True
    return dict(passed=passed and not retained, retained_evidence_to_review=retained, original=original['score'], variant=variant['score'], changes=changes)


def check(key, source, folder, staged=False, samples_only=False, indexed=False, reviewed=False, split=False, semantic=False):
    frozen = folder / key / '细则.json'
    if not frozen.exists():
        if semantic:
            from rubric_workshop.semantic_rubric import build, compact_minimum
            baseline = OUT / '最低充分边界复核' / key / '细则.json'
            if baseline.exists():
                base = load(baseline)
                if base['source'] != source:
                    raise ValueError('基础细则题源不一致')
                rubric = compact_minimum(source, base['rubric'])
            else:
                rubric = build(source, frozen.parent)
        elif split:
            from rubric_workshop.blind_batch import freeze_boundaries
            parent=load(OUT/'逐段反查回测'/key/'细则.json')
            if parent['source'] != source:
                raise ValueError('基础题源不一致')
            rubric=freeze_boundaries(source,parent['rubric'])
        elif staged:
            from rubric_workshop.staged_rubric import staged_rubric
            rubric = staged_rubric(source, frozen.parent, OUT/'逐步定位回测'/key if reviewed else None)
        else:
            rubric = core_rubric(source)
        save(frozen, dict(source=source, rubric=rubric))
    record = load(frozen)
    if record['source'] != source:
        raise ValueError('已冻结细则题源不同')
    if indexed:
        record['rubric'] = dict(record['rubric'], indexed_evidence=True)
        folder = folder / '句号取证复评'
    styles = {'概括稿':'用简洁的概括呈现各项实质。', '行动稿':'从具体行动或因果联系组织答案，选择有代表性的证据。'}
    samples = load(OUT / '高分参照库' / '样本.json')
    answers = {s['author']+str(s['version']): s['answer'] for s in samples if s['key']==key}
    if key in ('2023地市-1','2025地市-1'):
        answers.update({n[:-5]:a for n,a in cases_for(key).items() if '删' in n or '等义' in n})
    if semantic and key == '2023地市-1':
        independent = OUT / '核心条件与例证分列盲评' / key
        for name in ('概括稿', '行动稿'):
            answers['独立'+name] = load(independent / (name+'.json'))['answer']
        answers['套话稿'] = '坚持系统思维，凝聚各方合力，切实提升工作质效，推动高质量发展。'
        answers['概念误写稿'] = '把生态产品总值当作当年企业营业收入，直接给企业分红。'
    if samples_only:
        styles = {}
    errors = []
    for name in [*answers, *styles]:
        path = folder / key / (name + '.json')
        if path.exists():
            if name in answers and load(path)['answer'] != answers[name]:
                raise ValueError(f'已存答卷与输入不符：{name}')
            continue
        try:
            answer = write_answer(source, styles[name]) if name in styles else answers[name]
            marks = evaluate(source, record['rubric'], answer)
            save(path, dict(answer=answer, chars=count_pure_exam_chars(answer), marks=marks, score=sum(m['score'] for m in marks)))
            print(key, name, sum(m['score'] for m in marks), flush=True)
        except Exception as exc:
            errors.append(f'{name}：{exc}')
    baseline = next((folder/key/(s['author']+'1.json') for s in samples if s['key']==key and s['version']==1), None)
    if baseline and baseline.exists():
        contrasts = {}
        for name in answers:
            path = folder/key/(name+'.json')
            if name.startswith(key+'-') and path.exists():
                contrasts[name] = compare_answers(load(baseline),load(path),'等义' in name, allow_equal='删文旅' in name)
        if contrasts:
            save(folder/key/'对照验收.json', contrasts)
            errors.extend(f'{name}未通过对照' for name, result in contrasts.items() if not result['passed'])
    if reviewed:
        from rubric_workshop.staged_rubric import audit_exclusions
        path = frozen.parent/'排除项核验.json'
        if not path.exists():
            analysis=load(frozen.parent/'逐段复核.json')['analysis']
            save(path,audit_exclusions(source,analysis))
        errors.extend(f'{ref}遗漏独立意思：{reason}' for ref,(miss,_,reason) in load(path).items() if miss)
    if errors:
        raise ValueError(' / '.join(errors))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key', nargs='+')
    parser.add_argument('--staged', action='store_true')
    parser.add_argument('--samples-only', action='store_true')
    parser.add_argument('--indexed', action='store_true')
    parser.add_argument('--reviewed', action='store_true')
    parser.add_argument('--split', action='store_true', help='固定已复核细则的方面和权重，只细化部分得分条件')
    parser.add_argument('--semantic', action='store_true')
    args = parser.parse_args()
    if sum([args.staged,args.reviewed,args.split,args.semantic]) > 1:
        parser.error('请选择一种制标方法')
    keys = set(args.key or [s['key'] for s in load(OUT / '高分参照库' / '样本.json')])
    folder = OUT / ('独立概括与行动表达' if args.semantic else '保留方面权重拆分条件' if args.split else '逐段反查回测' if args.reviewed else '逐步定位回测' if args.staged else '核心条件与例证分列盲评')
    errors = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(check,k,s,folder,args.staged or args.reviewed,args.samples_only,args.indexed,args.reviewed,args.split,args.semantic):k for k,_,s in catalog() if k in keys}
        for job in as_completed(jobs):
            try:
                job.result()
            except Exception as exc:
                errors.append((jobs[job], str(exc)))
                print(*errors[-1], flush=True)
    raise SystemExit(bool(errors))
