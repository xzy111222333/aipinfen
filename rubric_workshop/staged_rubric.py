"""无机构答案制标。先保存任务和可独立给分的意思，再只给已通过检验的意思配分。"""
import re

from rubric_workshop.blind_batch import call_llm_json, count_pure_exam_chars, load, save


def make_map(source):
    instruction = '''阅读题面后，把材料按考生要完成的任务归并。先写 task（交付物、必要关系和概括层次）。返回 rows，每行六列 [分区,方面名,最短充分答句,删去该意思后具体漏掉哪一项任务,材料段号数组,该方面可以省去的细节]。同一方面的几个实例、机制与其直接结果如果共同说明同一意思，归在同一行；能独立回答题面另一层的才另起一行。答句是考生可以直接写的中文短句。组织项写可观察效果。逐段查遗漏，背景、实例、设想不能自动成为必答项。严格区分现状与将来。只输出形如 {"task":"一句话概括任务","rows":[["内容","名称","短答","遗漏的独立含义",["输入材料段号"],"可省的细节"]]} 的JSON；rows内只用数组。''' 
    payload = dict(source)
    for attempt in range(3):
        result = call_llm_json(instruction, payload)
        try:
            if not result['task'] or not isinstance(result['task'],(str,dict,list)):
                raise ValueError('任务须为一句话')
            names = set()
            claims = []
            for zone, name, claim, loss, refs, optional in result['rows']:
                if zone not in source['budget'] or name in names or not all(isinstance(v,str) and v.strip() for v in (name,claim,loss)):
                    raise ValueError('任务、名称或缺失含义无效')
                if not isinstance(refs,list) or (zone=='内容' and not refs) or any(ref not in source['materials'] for ref in refs):
                    raise ValueError('段号必须照抄输入键')
                names.add(name)
                if zone=='内容':
                    claims.append(claim)
            cap = int(re.search(r'不超过\s*(\d+)\s*字',source['question'])[1])
            size = count_pure_exam_chars('。'.join(claims))
            if not claims or size>cap or {r[0] for r in result['rows']} != set(source['budget']):
                raise ValueError(f'检查分区和字限，内容短答合计{size}格，上限{cap}格')
            result['content_chars']=size
            return result
        except (ValueError,TypeError,KeyError) as exc:
            if attempt==2:
                raise
            payload=dict(source, 上次结果=result, 修正=str(exc))


def review_map(source, draft):
    prompt = """独立复审任务图。逐段对照题干动词：哪些段落提供了一种独立回答，哪些只佐证已有方面、介绍背景或未来设想？归纳不能漏掉不同功能。对每项分别写考生可交卷的最低充分短句。可省去的名称、数字、机理不得嵌入短句。若同一方面两种做法能分别回答不同问题，分别列项；若是等价例证，只写其共同意思。区分义务与某次事件，只有材料确立的通则才是公文必须遵循的事实。输出 {"rows":[["内容","方面名","最短答句","少写后遗漏的意思",["材料段号"],"可省细节"]],"coverage":{"每一个输入材料段号":"方面名或排除理由"}}。简洁写全，预算分区和字限见题面。"""
    payload = dict(source, draft=draft['rows'])
    cap = int(re.search(r'不超过\s*(\d+)\s*字', source['question'])[1])
    for attempt in range(3):
        raw = call_llm_json(prompt,payload)
        try:
            if set(raw['coverage']) != set(source['materials']) or not all(isinstance(v,str) and v.strip() for v in raw['coverage'].values()):
                raise ValueError('逐段说明必须覆盖输入的全部材料段号')
            rows = raw['rows']
            names = set()
            for zone,name,claim,loss,refs,optional in rows:
                if name in names or zone not in source['budget'] or not all(isinstance(v,str) and v.strip() for v in (name,claim,loss)):
                    raise ValueError('重复项目或缺少必要意思')
                if not isinstance(refs,list) or (zone=='内容' and not refs) or not set(refs)<=set(source['materials']):
                    raise ValueError('材料依据无效')
                names.add(name)
            size=count_pure_exam_chars('。'.join(r[2] for r in rows if r[0]=='内容'))
            if size>cap or {r[0] for r in rows}!=set(source['budget']):
                raise ValueError(f'检查分区与字限，内容短答{size}格，上限{cap}格')
            return dict(raw,task=draft['task'],content_chars=size)
        except (KeyError,TypeError,ValueError) as exc:
            if attempt==2:
                raise
            payload=dict(source,draft=draft['rows'],校正=str(exc))


def staged_rubric(source, directory, base=None):
    path = (base or directory)/'任务与材料归类.json'
    if not path.exists():
        save(path,dict(source=source,analysis=make_map(source)))
    record=load(path)
    if record['source'] != source:
        raise ValueError('归类题源发生变化')
    analysis=record['analysis']
    if base is not None:
        reviewed = directory/'逐段复核.json'
        if not reviewed.exists():
            save(reviewed,dict(source=source,draft=analysis,analysis=review_map(source,analysis)))
        checkpoint=load(reviewed)
        if checkpoint['draft']!=analysis or checkpoint['source']!=source:
            raise ValueError('复核与原任务图版本不同')
        analysis=checkpoint['analysis']
    rows=analysis['rows']
    prompt='只按各项对题面任务的贡献分配分值。返回 JSON 的 points 数组，逐行给一个正数，是0.25的倍数；各区分别合计等于预算。不要改写或增补得分条件。'
    payload=dict(question=source['question'],budget=source['budget'],rows=[r[:4] for r in rows])
    for attempt in range(3):
        values=call_llm_json(prompt,payload).get('points')
        if isinstance(values,list) and len(values)==len(rows) and all(type(v) in (int,float) and v>0 and v*4%1==0 for v in values):
            totals={zone:sum(v for r,v in zip(rows,values) if r[0]==zone) for zone in source['budget']}
            if totals==source['budget']:
                break
        if attempt==2:
            raise ValueError(f'配分未通过，返回的分值为{values}')
        payload['上次分值']=values
        payload['修正']='完整返回数组，复算每区合计。'
    items=[]
    for (zone,name,claim,loss,refs,optional),weight in zip(rows,values):
        items.append(dict(zone=zone,name=name,points=weight,units=[[weight,'交付' if zone=='组织' else '语义',claim]],evidence={k:source['materials'][k] for k in refs},loss=loss,optional=optional))
    return dict(items=items,review_each_unit=True,limit=source['question'])


def audit_exclusions(source, analysis):
    """独立核对排除段，不把排除理由本身当作事实依据。"""
    used = {ref for row in analysis['rows'] for ref in row[4]}
    omitted = {k:v for k,v in source['materials'].items() if k not in used}
    if not omitted:
        return {}
    prompt = '核对未用段落。对照原文和题目，把已发生和拟发生分开。判断本段是否包含目前方面没有表达、却能独立回答题意的意思。返回 {"checks":{"输入段号":[true或false,"连续原文短引","判断理由"]}}。true指漏答，false表示已覆盖或可省。每段都须返回。'
    payload = dict(question=source['question'],materials=omitted,claims=[r[2] for r in analysis['rows']])
    from scoring.engine import is_verbatim_quote
    for attempt in range(3):
        raw=call_llm_json(prompt,payload)['checks']
        if isinstance(raw,dict) and set(raw)==set(omitted) and all(isinstance(v,list) and len(v)==3 and type(v[0]) is bool and is_verbatim_quote(v[1],omitted[k]) and isinstance(v[2],str) and v[2].strip() for k,v in raw.items()):
            return raw
        payload['修正']='每段给连续短引和是否遗漏的布尔值，覆盖所有输入段号。'
    raise ValueError('排除项审读缺少可核引文')
