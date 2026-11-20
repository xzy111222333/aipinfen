"""对首卷执行独立于制作提示词的边界试评。"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .__main__ import OUTPUT, grade, require

CASES = {
    1: (1, "协调上级部门新建园区铁路专用线，实现园区非临港钢铁企业铁路接入。协调港口、铁路和专用线降费。扩线增班，升级装卸设备。",
        "铁路价格下降，园区铁路货运占比提高。"),
    2: (1, "生态效益：藕鳖共养，甲鱼吃福寿螺护藕，排泄物肥田，种植使用沼液肥。",
        "经济效益：藕鳖共养，甲鱼吃福寿螺护藕，排泄物肥田，种植使用沼液肥。"),
    3: (2, "市民朋友们，S市科委发布公众参与科学项目。项目包含鸟撞调查、昆虫家谱，旨在提升科学素养，让市民参与收集数据、支持科研。请登录公众参与科学项目平台注册报名，按项目参加培训，进行观察、记录并上传。欢迎大家用身边的发现，为科研添一份力量！",
        "向领导汇报：S市科委发布公众参与科学项目。项目包含鸟撞调查、昆虫家谱，旨在提升科学素养，让市民参与收集数据、支持科研。市民在公众参与科学项目平台注册报名，按项目参加培训，进行观察、记录并上传。以上，请领导审阅。"),
    4: (2, "部分出租户担心房屋收不回，承租人又担心租期届满后无法收回改造投入。统一合同，明确交还、续租和改造权益。燃气入户涉及多个部门，审批长期搁置。县里明确牵头单位，对申请实行部门联办、限时反馈。",
        "要完善服务、建设美丽乡村、促进群众共同参与。"),
    5: (3, "环境考古研究需要把不同学科的证据拼到同一问题上。单看骨骼，研究者能推测个体特征，却难以解释人群为什么迁移。引入地理和环境证据，可以把迁移与生存条件联系起来，进而提出单一方向难以发现的假设。多样性要经过共同提问和相互检验，才能产生新的解释。",
        "考古研究的核心是提高投资力度。实验设备越贵，研究水平就越高。要多投入资金，扩大实验室规模，采购更先进的仪器，再把资金投向成果产出更高的单位。只要增加投入，原创成果自然会出现。"),
}


def verify(number, directory):
    path = directory / f"第{number}题.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    kind, better, weaker = CASES[number]
    require(saved["状态"] in ("已锁定", "人工核对") and saved["分析"]["题型"] == kind, "题型或状态错误")
    marks = grade(saved["题目材料"], saved["分析"], saved["细则"], [better, weaker])
    scores = [mark["得分"] for mark in marks]
    expected = scores[0] > scores[1]
    if saved["状态"] == "人工核对" and number == 2:
        expected = scores == [4, 0]
    if number == 3:
        content = {row["编号"] for row in saved["细则"]["评分项"] if row["类别"] == "内容"}
        subtotals = [sum(item["分值"] for item in mark["明细"] if item["编号"] in content)
                     for mark in marks]
        expected = expected and subtotals[0] == subtotals[1]
    return {"题号": number, "分数": scores, "符合预期": expected,
            "答卷": [better, weaker], "判卷": marks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("numbers", type=int, nargs="*")
    parser.add_argument("--directory", type=Path, default=OUTPUT)
    args = parser.parse_args()
    numbers = args.numbers or list(CASES)
    require(all(number in CASES for number in numbers), "题号只能在 1 到 5 之间")

    def run(number):
        try:
            return verify(number, args.directory)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            return {"题号": number, "符合预期": False, "错误": str(error)}

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(run, numbers))
    path = args.directory / "独立边界试评.json"
    prior = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    combined = {row["题号"]: row for row in prior + results}
    path.write_text(json.dumps([combined[key] for key in sorted(combined)],
                               ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{"题号": row["题号"], "分数": row.get("分数"), "符合预期": row["符合预期"],
                       "错误": row.get("错误")}
                      for row in results], ensure_ascii=False))
    return int(not all(row["符合预期"] for row in results))


if __name__ == "__main__":
    sys.exit(main())
