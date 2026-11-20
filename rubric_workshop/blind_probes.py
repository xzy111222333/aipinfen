"""盲测已冻结题的局部判分边界；试评看不到期望值。"""
from rubric_workshop.blind_batch import OUT, call_llm_json, load, save
from concurrent.futures import ThreadPoolExecutor
import hashlib
import random

def check(f):
    path=OUT/"边界盲测"/f.name
    digest=hashlib.sha256(f.read_bytes()).hexdigest()
    if path.exists() and load(path)["hash"]==digest:
        return
    frozen=load(f)
    probes=frozen["audits"][-1]["probes"]
    order=list(range(len(probes)))
    random.Random(f.stem).shuffle(order)
    submitted=[dict(answer=probes[i]["answer"],target=probes[i]["target"]) for i in order]
    response=call_llm_json("按冻结分表独立评每条局部答句，仅评指定项目，按输入顺序输出 marks 数组，每项只含数字 score 和简短 reason。不因同批其他答句而改分。",dict(question=frozen["source"]["question"],items=frozen["rubric"]["items"],cases=submitted))
    marks=response if isinstance(response,list) else response["marks"]
    if len(marks)!=len(order): raise ValueError(f.name+"边界条数不符")
    outcomes=[]
    for i,m in zip(order,marks):
        expected=probes[i]["expected"]
        outcomes.append(dict(case=probes[i],observed=m,passed=m["score"]==expected))
    save(path,dict(hash=digest,results=outcomes))
if __name__=="__main__":
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(check,sorted((OUT/"锁定").glob("*.json"))))
