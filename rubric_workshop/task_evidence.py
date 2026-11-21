"""制标前按题干冻结材料用途，再沿用原有逐项或档位流程。"""
from rubric_workshop.blind_batch import call_llm_json, load, make, save


TASKS = '''只读题干与指定材料。把需要读者得到的独立意思作为行。题干引文逐字短摘，任务可组合。每行写明删除后读者少知道什么；替代案例列在同一行。区分事实复述、可行建议、面向读者的行动指引和用事例支持判断。评论分别列材料事实、事实说明什么和可替代角度。不同的给分行不可重复拥有同一语义。输出 {"rows":[{"task":"题干连续原文","meaning":"必要意思","sources":["原段号"],"optional":"可省略的例证或无","missing":"只剩空泛词时缺少的关系"}]}。'''


def build(source, directory):
    path = directory / '任务与证据.json'
    if path.exists():
        record = load(path)
        if record['source'] != source:
            raise ValueError('证据表题源已变化')
        rows = record['rows']
    else:
        payload = dict(source)
        for attempt in range(3):
            rows = call_llm_json(TASKS, payload).get('rows')
            if isinstance(rows, list) and rows and all(isinstance(row, dict) and
                    all(isinstance(row.get(k), str) and row[k].strip() for k in ('task','meaning','missing','optional')) and
                    row['task'] in source['question'] and isinstance(row.get('sources'), list) and
                    row['sources'] and all(p in source['materials'] for p in row['sources']) for row in rows):
                break
            payload['校正'] = '每行题干引文必须连续照抄；sources必须使用至少一个输入材料键。'
        else:
            raise ValueError('任务证据表存在不可追溯的行')
        save(path, dict(source=source, rows=rows))
    rubric = make(source, dict(任务证据=rows, 执行='在分配分值前核对每项独立缺失的意思。将等价内容合并。建议按能否解决具体问题且符合材料约束评分。短评定档要写明事实如何支持评价，孤立罗列事实和只点出对象均不足以证明评析充分。'))
    return dict(rubric, task_evidence=True)


def compile_units(source, draft):
    """将审阅后的独立语义与分值绑定，防止阅卷时自行拆分分数。"""
    if source['type'] == 3:
        return draft
    prompt = '''把细则草案重新编成可独立给分的最少判据。题干是取舍依据，可重组草案。每个判据只代表一个足以给分的意思，概括和实例能回答同一个问题时并列为两条满分表达路径。一条措施的远端效益仅在题干需要时独立计分。一个动作在多个段落或要求中出现，只归一项。判据不可出现必须一口气答出多个可分离动作才能得分。篇幅是约束，不因未超字自动送分。组织只奖励面向整篇的可观察效果，孤立一条句子不能证明全文全面有条理。建议允许材料约束内的其他可行措施。给每项整数或0.25的倍数分值，各区精确加总；逐项返回 [预算区名,简明且唯一的项目名,满分,得分的最低充分意思,等价充分表达,易混但不足的意思,相关材料段号数组]。含组织项时段号可以为空。只输出 {"units":[...]}，允许用证据直接实现同一上位意思。'''
    if source['type'] == 1:
        prompt += '第一类只有材料内容分。草案如把全篇条理、格式或篇幅列入内容，移除对应行并将分值分配给独立内容意义。条理体现为读者可以识别不同实质，不能占据独立内容分。'
    payload = dict(题源=source, 草案=draft['items'], 允许段号=list(source['materials']))
    for attempt in range(4):
        try:
            units = call_llm_json(prompt, payload)['units']
            totals = dict.fromkeys(source['budget'], 0)
            names = set()
            items = []
            for zone,name,points,full,equiv,near,refs in units:
                if zone not in totals or name in names or not all(isinstance(t,str) and t.strip() for t in (name,full,equiv,near)):
                    raise ValueError('项目命名或分区错误')
                if type(points) not in (int,float) or points<=0 or points*4 % 1:
                    raise ValueError('分数须为正的四分之一分倍数')
                if not isinstance(refs,list) or (zone=='内容' and not refs) or not set(refs)<=source['materials'].keys():
                    raise ValueError(f'{name} 的段号 {refs} 无效。请原样复制允许段号；组织项用空数组。')
                totals[zone] += points
                names.add(name)
                items.append(dict(zone=zone,name=name,points=points,full=full,equivalent=equiv,near=near,paragraphs=refs))
            if totals != source['budget']:
                raise ValueError(f'目前各区合计{totals}，应为{source["budget"]}')
            return dict(items=items, task_evidence=True, binary_units=True, limit=draft['limit'])
        except (KeyError, TypeError, ValueError) as error:
            if attempt == 3:
                raise
            payload['校正'] = str(error)
            payload['上轮判据'] = locals().get('units', [])


def grade_units(source, rubric, answer):
    import re
    sentences = [s.strip() for s in re.findall(r'[^。！？；\n]+[。！？；]?', answer) if s.strip()]
    units = {str(i): {k:item[k] for k in ('zone','name','full','equivalent','near')} for i,item in enumerate(rubric['items'],1)}
    prompt = '''逐项独立判断整份匿名答卷是否明确支持该最低意思。读上下文处理否定、矛盾和材料范围。概括或另一个逻辑等价的做法均可；同一原因不可重复扣。返回 {"decisions":{"1":[true,[句号序号],"证据怎样满足条件"],"2":[false,[],"具体缺失的意思"]}}。每个编号均需判定；先确定证据再判断，不参考其他分项，不估计总分。'''
    from scoring.grid_sheet import count_pure_exam_chars
    payload = dict(题源=source,判据=units,字数=count_pure_exam_chars(answer),答卷=dict(enumerate(sentences,1)))
    for attempt in range(3):
        try:
            decisions=call_llm_json(prompt,payload)['decisions']
            if decisions.keys()!=units.keys():
                raise ValueError('编号须覆盖全部判据')
            marks=[]
            for key,item in zip(units,rubric['items']):
                hit,refs,why=decisions[key]
                if type(hit) is not bool or not isinstance(refs,list) or any(type(n) is not int or not 1<=n<=len(sentences) for n in refs) or hit and not refs or not isinstance(why,str) or not why.strip():
                    raise ValueError(f'判据{key}需要有效答卷证据或缺失理由')
                quotes=[sentences[n-1] for n in refs]
                marks.append(dict(name=item['name'],score=item['points'] if hit else 0,quote=quotes[0] if quotes else '',citations=quotes,why=why))
            return marks
        except (KeyError,TypeError,ValueError) as error:
            if attempt==2:
                raise
            payload['校正']=str(error)
