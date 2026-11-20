"""将冻结细则、逐项分数和低分复查整理为可读的逐题索引。"""
from collections import Counter
from statistics import median
from rubric_workshop.blind_batch import OUT, catalog, load, save, call_llm_json, is_verbatim_quote
from concurrent.futures import ThreadPoolExecutor

LOW = 0.6

def readable(key, src, rubric, path):
    text = [f"# {key} 评分规则与细则", "", src["question"], "",
            f"计分类型 {src['type']}。分区预算 " +
            "、".join(f"{k} {v:g} 分" for k,v in src["budget"].items()) + "。", ""]
    for i in rubric["items"]:
        text.extend([f"## {i['name']}（{i['points']} 分）", "",
                     str(i["levels"]), "", f"可接受表达　{i['accept']}",
                     "", f"易错表达　{i['reject']}", ""])
        for para, quote in i["evidence"].items():
            text.extend([f"- {para}　{quote}", ""])
    text.extend(["## 字数与作答边界", "", str(rubric["limit"]), "",
                 "## 各段材料去向", ""])
    for para in src["materials"]:
        text.extend([f"- {para}　{rubric['paragraphs'][para]}", ""])
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("\n".join(text), encoding="utf8")

def diagnose(entry,baseline=False):
    key, _, src = entry
    file = OUT/("首轮试评" if baseline else "试评")/(key+".json")
    if not file.exists():
        return
    evaluation = load(file)
    total = sum(src["budget"].values())
    path = OUT/("首轮低分复查" if baseline else "低分复查")/file.name
    reviews = []
    if path.exists():
        old=load(path)
        if old["hash"]==evaluation["hash"] and old.get("schema")==2:
            return
    for row in evaluation["answers"]:
        if row["score"] >= LOW*total:
            continue
        frozen = load(OUT/("首轮冻结" if baseline else "锁定")/file.name)
        for _ in range(3):
            raw=call_llm_json(
                """只做低分诊断。分别检查细则过严、实际阅卷错误、答卷缺漏或录入异常，不预设任何一方对。选最有影响的1至3项，返回 {"diagnosis":[{"cause":"归因","item":"评分项","answer_quote":"答卷中一处连续短原话","material_quote":"材料中一处连续短原话","reason":"同时判断规则与给分是否合理"}]}。引文不超过15字，必须逐字连续，不得用省略号或自己归纳。不写示范答案。""",
                dict(题目=src,细则=frozen["rubric"]["items"],试评=row))
            findings=raw if isinstance(raw,list) else raw.get("diagnosis",[])
            if findings and all(isinstance(d,dict) and all(d.get(k) for k in ("cause","item","reason")) and is_verbatim_quote(d.get("answer_quote"),row["answer"]) and any(is_verbatim_quote(d.get("material_quote"),v) for v in src["materials"].values()) for d in findings):
                reviews.append(dict(diagnosis=findings))
                break
        else: raise ValueError(key+" 低分诊断未返回可核验引文")
    if reviews:
        save(path,dict(hash=evaluation["hash"],schema=2,reviews=reviews))

def report():
    entries=list(catalog())
    lines=["# 五年国考小题无参考制标与试评","",
           "每行对应一个题位。同卷异题及跨卷复用均按题位列示。同题多份署名答案分别评阅；低分诊断阈值事先设为满分的60%以下，只用于排查。中位数先按同题多个版本取中位，再在题位间取中位。","",
           "|题位|类型|满分|评分细则|白鹭首轮冻结试评|复核版试评|","|---|---:|---:|---|---|---|"]
    counts=Counter()
    ratios=[]
    for key,_,src in entries:
        name=key+".json"
        total=sum(src["budget"].values())
        frozen=OUT/"锁定"/name
        draft=OUT/"初制"/name
        review=OUT/"待复核"/name
        status="缺少细则"
        score="待试评"
        if draft.exists():
            counts["初制"]+=1
            readable(key,src,load(draft),OUT/"初制细则"/(key+".md"))
            status=f"[初制](初制细则/{key}.md)"
        if review.exists() and not frozen.exists():
            status+=f" [待复核](待复核/{name})"
        if frozen.exists():
            readable(key,src,load(frozen)["rubric"],OUT/"逐题细则"/(key+".md"))
            status=f"[逐项细则](逐题细则/{key}.md) · [冻结记录](锁定/{name})"
            counts["锁定"]+=1
        grade=OUT/"试评"/name
        if grade.exists():
            rows=load(grade)["answers"]
            counts["试评版本"]+=len(rows)
            if rows:
                counts["有试评题位"]+=1
                scores=[r["score"] for r in rows]
                ratios.append(median(scores)/total)
                counts["低分版本"]+=sum(s<total*LOW for s in scores)
                counts["含低分题位"]+=int(min(scores)<total*LOW)
                score=f"[{', '.join(str(s) for s in scores)}](试评/{name})"
                if (OUT/"低分复查"/name).exists():
                    score+=f" [诊断](低分复查/{name})"
            else:
                score="无独立白鹭署名版本"
        first=OUT/"首轮试评"/name
        first_score="待试评"
        if first.exists():
            variants=load(first)["answers"]
            first_score="无独立白鹭署名版本"
            if variants:
                counts["首轮已评题位"]+=1
                counts["首轮已评版本"]+=len(variants)
                scores=[r["score"] for r in variants]
                counts["首轮低分版本"]+=sum(s<total*LOW for s in scores)
                counts["首轮含低分题位"]+=int(min(scores)<total*LOW)
                first_score=f"[{', '.join(str(s) for s in scores)}](首轮试评/{name})"
                if (OUT/"首轮低分复查"/name).exists():
                    first_score+=f" [诊断](首轮低分复查/{name})"
        lines.append(f"|{key}|{src['type']}|{total:g}|{status}|{first_score}|{score}|")
    lines[2:2]=["；".join(f"{k} {v}" for k,v in counts.items()) + (f"；题位得分率中位数 {median(ratios):.1%}。" if ratios else "。"),""]
    (OUT/"结果索引.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print(counts,flush=True)

if __name__=="__main__":
    with ThreadPoolExecutor(6) as pool:
        list(pool.map(diagnose,catalog()))
        list(pool.map(lambda entry: diagnose(entry,True),catalog()))
    report()
