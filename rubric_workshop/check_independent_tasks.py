"""回放冻结后答卷。python -m rubric_workshop.check_independent_tasks"""
import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from rubric_workshop.blind_batch import OUT, catalog, count_pure_exam_chars, evaluate, load, save
from rubric_workshop.semantic_rubric import build, review_independent_tasks

BASE = OUT / '独立概括与行动表达'
TRIAL = OUT / '删除检验跨题回测'
HOLDOUT = '2022地市-'


def prepare(entry):
    key, path, source = entry
    dest = TRIAL / key / '细则.json'
    if not dest.exists():
        prev = BASE / key / '细则.json'
        if prev.exists():
            record = load(prev)
            if record['source'] != source:
                raise ValueError(f'{key}题源变更')
            parent = record['rubric']
        else:
            parent = build(source, dest.parent)
        save(dest, dict(source=source, rubric=review_independent_tasks(source, parent)))
    frozen = load(dest)
    if frozen['source'] != source or not frozen['rubric'].get('independent_tasks'):
        raise ValueError(f'{key}冻结表不匹配')
    if key.startswith(HOLDOUT):
        texts = re.findall(r'^### ([^\n]+)\n(.*?)(?=^### |\Z)', path.read_text(encoding='utf8'), re.M|re.S)
        answers = list(dict.fromkeys(text.strip() for name,text in texts if '白鹭' in name and '改进' not in name))
        if not answers:
            raise ValueError(f'{key}缺留出答卷')
        jobs = [(key, f'白鹭{idx}', text, frozen) for idx,text in enumerate(answers,1)]
        if key == '2022地市-1':
            answer = answers[0]
            start = answer.index('3、H铁路局')
            end = answer.index('4、C市',start)
            jobs.append((key,'删第三案例',answer[:start]+answer[end:],frozen))
        return jobs
    answers = []
    for file in sorted((BASE / key).glob('*.json')):
        record = load(file)
        if 'marks' in record and 'answer' in record:
            answers.append((key,file.stem,record['answer'],frozen))
    return answers


def grade(job):
    key, name, answer, record = job
    dest = TRIAL / key / (name+'.json')
    if dest.exists():
        if load(dest)['answer'] != answer:
            raise ValueError(f'{key}/{name}答卷有变')
        return
    marks = evaluate(record['source'],record['rubric'],answer)
    score = sum(m['score'] for m in marks)
    save(dest,dict(answer=answer,chars=count_pure_exam_chars(answer),marks=marks,score=score))
    print(key,name,score,flush=True)


def audit_controls():
    checks = []
    for key, original, variant, target in [
        ('2023地市-1', '袁东1', '2023地市-1-等义转述', None),
        ('2023地市-1', '袁东1', '2023地市-1-删融资', '创新GEP贷融资'),
        ('2025地市-1', '白鹭1', '2025地市-1-删文旅', None),
        ('2025地市-1', '白鹭1', '2025地市-1-删储碳', '碳库作用'),
    ]:
        folder = TRIAL/key
        a,b = [load(folder/(name+'.json')) for name in (original,variant)]
        items = load(folder/'细则.json')['rubric']['items']
        changes = []
        valid = True
        for item,left,right in zip(items,a['marks'],b['marks'],strict=True):
            if left['name'] != item['name'] or right['name'] != item['name']:
                raise ValueError('答卷与冻结规则不对应')
            if left['score'] != right['score']:
                changes.append([item['name'],left['score'],right['score']])
                valid &= target in item.get('source_items',[]) and left['score'] > right['score']
        checks.append(dict(对照=variant,通过=bool(valid and (target is None or changes)),分项变化=changes))
    for name in ('概念误写稿','套话稿'):
        score = load(TRIAL/'2023地市-1'/(name+'.json'))['score']
        checks.append(dict(对照=name,通过=score==0,得分=score))
    folder = TRIAL/'2022地市-1'
    original, variant = [load(folder/(n+'.json')) for n in ('白鹭1','删第三案例')]
    changes = [(a['name'],a['score'],b['score']) for a,b in zip(original['marks'],variant['marks'],strict=True) if a['score']!=b['score']]
    checks.append(dict(对照='留出卷删第三案例',通过=len(changes)==1 and '工业' in changes[0][0] and changes[0][1]>0 and changes[0][2]==0,分项变化=changes))
    save(TRIAL/'定向对照.json',checks)
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key',nargs='+')
    args=parser.parse_args()
    entries=[e for e in catalog() if e[2]['type'] in (1,2) and ((BASE/e[0]/'细则.json').exists() or e[0].startswith(HOLDOUT)) and (not args.key or e[0] in args.key)]
    errors=[]
    with ThreadPoolExecutor(max_workers=6) as makers, ThreadPoolExecutor(max_workers=8) as graders:
        making={makers.submit(prepare,e):e[0] for e in entries}
        grading={}
        for job in as_completed(making):
            try:
                for entry in job.result():
                    grading[graders.submit(grade,entry)] = entry[:2]
            except Exception as exc:
                errors.append((making[job],str(exc)))
        for job in as_completed(grading):
            try:
                job.result()
            except Exception as exc:
                errors.append((grading[job],str(exc)))
    for error in errors:
        print(error,flush=True)
    if not errors and not args.key:
        checks = audit_controls()
        print('对照通过',sum(row['通过'] for row in checks),'/',len(checks))
        return int(any(not row['通过'] for row in checks))
    return int(bool(errors))

if __name__ == '__main__':
    raise SystemExit(main())
