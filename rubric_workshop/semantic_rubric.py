"""用段落事实和可直接交卷的最低充分意思制作小题细则。"""
import re
from rubric_workshop.blind_batch import call_llm_json, count_pure_exam_chars, load, save


def draft(source, directory):
    paragraphs = list(source['materials'].items())
    path = directory/'逐段事实.json'
    if path.exists():
        record = load(path)
        if record['source'] != source:
            raise ValueError('题源已变更')
        facts = record['facts']
    else:
        payload = dict(题目=source['question'], 段落=[text for _,text in paragraphs])
        prompt = '按输入段落顺序逐段提取原文对题目的贡献，每段一句短话。保留材料自己的总括和关系、主体、已然或未然、个案或通则；区分做法与结果。返回 {"facts":["第一段事实","第二段事实"]}，包括背景段，与输入逐段一一对应。'
        for attempt in range(3):
            facts = call_llm_json(prompt,payload).get('facts')
            if isinstance(facts,list) and len(facts)==len(paragraphs) and all(isinstance(x,str) and x.strip() for x in facts):
                break
            payload['校正']=f'共{len(paragraphs)}段，须依次返回等长的字符串数组。'
        else:
            raise ValueError('逐段事实数量不符')
        save(path,dict(source=source,facts=facts))
    cap = int(re.search(r'不超过\s*(\d+)\s*字',source['question'])[1])
    prompt = '''制定这道题的方面分档细则。先读题干的动词和交付物，再读各段事实与原文。问哪些方面时优先材料明确概括的功能、关系；问如何实现时保留关键动作及作用，公文考虑读者的行动需求。每个方面写一个考生可以直接交卷的最低充分短答，再写缺少一层但仍做对一部分的短答、另一种充分表达，以及易混错误。最短答句不可堆材料细节；互相替代的例证不能分别占分。多项独立任务要拆开。保留事实的时态和适用范围，近错例写出何种错误会使本项只能部分得分或不得分。组织分用可观察的读者效果。给每项1至3的相对重要性，3表示主要任务。返回JSON {"rows":[["内容或组织","方面名",重要性,"最低充分短答","部分短答","等价短答","近错或空泛短答",[1,2]]],"omit":[不承担独立分值且未在任何行标注的段号]}。段号从1开始。内容短答合计须在限字内。将有同等功能的案例归入对应行的段号。'''
    payload=dict(题目=source['question'],预算=source['budget'],原文及独立摘要={str(i):[text,fact] for i,((_,text),fact) in enumerate(zip(paragraphs,facts),1)})
    prompt += '每一行的第一个字段只允许预算中的键：' + '、'.join(source['budget']) + '。只有内容预算时，所有行均填内容，禁止生成组织行。重要性字段用整数。'
    for attempt in range(3):
        result=call_llm_json(prompt,payload)
        try:
            items=[]
            all_ids=set(range(1,len(facts)+1))
            omitted=set(result['omit'])
            used=set()
            names=set()
            for zone,name,rank,claim,partial,alternative,reject,refs in result['rows']:
                if zone not in source['budget'] or name in names or type(rank) is not int or rank not in (1,2,3) or not all(isinstance(t,str) and t.strip() for t in (name,claim,partial,alternative,reject)):
                    raise ValueError(f'无效评分行：分区{zone}，合法分区{list(source["budget"])}，名称{name}，重要性{rank}；名称不得重复，四条短答均须为非空字符串')
                if not isinstance(refs,list) or (zone=='内容' and not refs) or not set(refs)<=all_ids:
                    raise ValueError('核对材料段号')
                names.add(name)
                used.update(refs)
                items.append(dict(zone=zone,name=name,rank=rank,claim=claim,partial=partial,alternative=alternative,reject=reject,evidence=dict(paragraphs[i-1] for i in refs)))
            if used|omitted!=all_ids or used&omitted:
                raise ValueError(f'未归类段号{all_ids-used-omitted}，重复排除{used&omitted}')
            size=count_pure_exam_chars('。'.join(i['claim'] for i in items if i['zone']=='内容'))
            for zone,budget in source['budget'].items():
                group=[i for i in items if i['zone']==zone]
                if not group or budget*4 < len(group):
                    raise ValueError('分区不能分配')
                # 四分之一分为单位，按重要性分配余数，保证加总精确。
                units=int(budget*4)-len(group)
                total=sum(i['rank'] for i in group)
                shares=[divmod(units*i['rank'],total) for i in group]
                left=units-sum(q for q,_ in shares)
                extra=set(sorted(range(len(group)),key=lambda n:-shares[n][1])[:left])
                for n,item in enumerate(group):
                    item['points']=(1+shares[n][0]+int(n in extra))/4
            return dict(items=items,aspect_levels=True,indexed_levels=True,content_chars=size,omitted=[paragraphs[i-1][0] for i in sorted(omitted)],limit=source['question'])
        except (KeyError,TypeError,ValueError) as exc:
            if attempt==2:
                raise
            payload['校正']=str(exc)


def review_minimum(source, rubric):
    items = rubric['items']
    fields = ('claim', 'partial', 'alternative', 'optional', 'boundary')
    prompt = """你在细则冻结前复核每项最低充分语义。只依据题面和原材料，保持方面和权重。先删掉草案中年份、专名、列举、规模数字等具体信息，问删后还完整回答本问吗？可以删的列入optional。题干需要精确对象、条件、数值或专门概念时保留，在boundary解释删后漏答的实质。问成效允许概括已实现的变化，问如何做须保留区分做法的动作及对象，公文兼顾受众用途与事实边界。组织项按可观察的整体效果。逐项输出编号映射 {"rows":{"1":["足以满分的简短意思","只答对一部分的意思","不同表达路径的充分短答","可省略或互相替代的细节","区别于套话或近错的必要语义及理由"]}}。每项完整答句应能直接用于答卷；部分档不可与完整档等义。"""
    rules = {str(n): [item['zone'], item['name'], item['claim'], item['partial'], list(item['evidence'])]
             for n, item in enumerate(items, 1)}
    payload = dict(题目=source['question'], 材料=source['materials'], 草案=rules)
    cap = int(re.search(r'不超过\s*(\d+)\s*字', source['question'])[1])
    for attempt in range(3):
        try:
            rows = call_llm_json(prompt, payload)['rows']
            if set(rows) != set(rules):
                raise ValueError('复核项与草案不一一对应')
            revised = []
            for n, item in enumerate(items, 1):
                values = rows[str(n)]
                if not isinstance(values, list) or len(values) != len(fields) or not all(isinstance(v, str) and v.strip() for v in values):
                    raise ValueError('每项须返回五条非空判据')
                revised.append(dict(item, **dict(zip(fields, values))))
            size = count_pure_exam_chars('。'.join(item['claim'] for item in revised if item['zone'] == '内容'))
            return dict(rubric, items=revised, content_chars=size, reviewed_minimum=True)
        except (KeyError, TypeError, ValueError) as exc:
            if attempt == 2:
                raise
            payload['校正'] = str(exc)


def build(source, directory):
    path = directory / '方面草案.json'
    if path.exists():
        record = load(path)
        if record['source'] != source:
            raise ValueError('草案与当前题源不一致')
    else:
        record = dict(source=source, rubric=draft(source, directory))
        save(path, record)
    return compact_minimum(source, review_minimum(source, record['rubric']))


def compact_minimum(source, rubric):
    items = rubric['items']
    prompt = """给阅卷人写可执行的最小给分条件。先逐字看题目问的对象和动词，然后对每项给两条各自独立即可得满分的答句：一条直接概括该项关系，一条用代表性的行动体现相同关系。每条只写一个完整核心关系，尽量在22格内；必要时可超出。问配合就写谁回应谁、怎样回应；问做法就写动作对象；问成效就写改变了什么；问并列任务时只保留该项负责的任务。删去不改变这层意思的原因、过程、效果与例子；题干明问的专门对象或条件须保留。第三条是仅表达一部分的答句，第四条是容易混淆的错误答句。组织项写可从全篇观察的文种或读者效果。返回 {"routes":{"1":["概括满分","行动满分","部分","零分"]}}。题目和材料是唯一事实依据。"""
    keys = [str(n) for n in range(1, len(items) + 1)]
    payload = dict(题目=source['question'], 材料=source['materials'], 方面={k:[i['zone'],i['name'],list(i['evidence'])] for k,i in zip(keys,items)})
    cap = int(re.search(r'不超过\s*(\d+)\s*字', source['question'])[1])
    for attempt in range(3):
        try:
            result = call_llm_json(prompt, payload)['routes']
            if set(result) != set(keys):
                raise ValueError('方面编号不齐')
            revised = []
            for k,item in zip(keys, items):
                row = result[k]
                if not isinstance(row,list) or len(row)!=4 or not all(isinstance(x,str) and x.strip() for x in row):
                    raise ValueError(f'核对第{k}项四条短答')
                revised.append(dict(item, claim=row[0], alternative=row[1], partial=row[2], reject=row[3]))
            size = count_pure_exam_chars('。'.join(i['claim'] for i in revised if i['zone']=='内容'))
            if size>cap:
                raise ValueError('最短充分表达累计超过字限')
            return dict(rubric,items=revised,content_chars=size,compact_minimum=True)
        except (KeyError,TypeError,ValueError) as exc:
            if attempt==2:
                raise
            payload['校正']=str(exc)

