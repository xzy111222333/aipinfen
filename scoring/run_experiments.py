import concurrent.futures
import json
import os
import time
from pathlib import Path

from scoring.engine import (
    align_reference_answers,
    compile_quantitative_rubric,
    extract_verified_cards,
    grade_answer_with_audit,
    load_question_bundle,
    parse_hard_constraints,
    SPEC_PROMPT,
    call_llm_json,
)


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "预制与实测"


Q3_SYNTHETIC_ANSWERS = {
    "模拟考生1_大白话口语务实卷": """关于2026年老百姓参与科学项目的发布词
街坊邻居、市民朋友们：
大家好！为了在咱们市里营造爱科学、尊重科学的好风气，提高大家的科学素养，给知识创新添把火，市科委联合高校和科研单位，打算在2026年1月推出8个“公众参与科学项目”。以前项目主要研究生态环境，今年还要拓展到天文学、心理学好几个新领域！
这种“专业科学家+普通市民一起干”的模式，干的是数据收集、技术探索这些真活儿，完全不挑年龄和工作——不管您是木匠、程序员还是小学生都能来。参加进来，咱们普通人能长知识、交到志同道合的朋友，有机会评上优秀参与者、拿到“公众科学家”的酷炫身份，还能实打实帮科学家大忙、出新点子！
参加办法特别简单：项目按“发布-注册-培训-观察记录-上传数据-核验-出结论”一步步走。大家直接上“公众参与科学项目”网络平台注册报名，看看在线教程、直播课或者线下讲座学会怎么用设备和做记录，就能利用周末空闲去观察上传数据，最后科学家核验后会发布正式报告。
乡亲们、朋友们，别犹豫啦，赶紧登录线上平台报名参加吧！
S市科委
2026年1月""",

    "模拟考生2_狂抄四个故事细节漏主干卷": """公众参与科学项目发布词
广大市民朋友：
10岁的豆豆以前觉得科学遥不可及，妈妈带他参加科普讲座和貉口普查后觉得离科学近了。木匠许师傅靠自学的昆虫摄影技术参加活动，像海绵一样吸收很多知识养分，还被评为优秀参与者。软件工程师小郭第三次担任普查领队，享受公众科学家的隐藏身份，觉得科学家一点不高冷，很务实谦逊。
2019年章教授团队买了80多台红外照相机和市民布设在绿地公园，拍到国家一级保护动物小灵猫因施工迁移，科研人员预判小河阻隔就联合绿化部门架设临时浮桥帮小灵猫安全渡河找到新栖息地。2023年夏天300多位市民在周末夜里打着手电筒调查貉的数量，记录猫粮投喂和垃圾管理，只用两个晚上就查完90个小区，顶得上传统科研团队两个半月的工作量！2024年12月发布的《S市昆虫名录》物种从3090种涨到3556种，20%的新增记录都是公众科学家发现的！
科学真有趣，欢迎大家关注科学！""",

    "模拟考生3_热血排比鸡汤空喊口号卷": """点燃科学火炬 共筑创新未来！
亲爱的市民朋友们：
科学是照亮人类文明进步的璀璨灯塔！创新是驱动城市蓬勃发展的澎湃引擎！在这个日新月异的新时代，S市科委隆重发布2026年公众参与科学项目，吹响全民探索未知的嘹亮号角！
为什么要参与科学？因为科学就在每一次仰望星空的惊叹里，在每一次俯身大地的追寻中！只要心怀热爱，每个人都是改变世界的火种；只要迈出脚步，平凡的双手也能托举伟大的奇迹！让我们打破心中的犹豫，燃烧求知的热血，用双手触摸真理的温度，用智慧点亮城市的星空！
听吧，科学的召唤已经响彻云霄！看吧，探索的大门已经为您敞开！广大市民朋友们，让我们携手并肩、踊跃报名，立刻投身这场波澜壮阔的科学盛宴，共同书写属于S市的辉煌科技篇章吧！！！
S市科委
2026年1月""",

    "模拟考生4_受众串台只招科学家申报卷": """2026年公众参与科学项目发布词
各位科研工作者与社会同仁：
为营造崇尚科学、尊重科学的浓厚氛围，提升市民科学素养，为知识生产注入新动能，S市科委大力推动“专业科研+公众参与”科研模式，目前正在筹备于2026年1月发布8个新项目，领域将从生态学扩展至天文学、心理学等多个方向。
公众科学模式涵盖数据收集、技术探索、问题研究，按照“项目发布-公众注册-项目培训-观察记录-数据上传-处理校验-研究结论”的流程推进，科研团队负责专业科研问题设计、任务拆解、课题项目发布及多源数据验证并形成研究报告。
在此，我委全力支持各领域的科学家积极申报“公众参与科学项目”，项目一经立项后将直接予以专项经费支持！诚挚呼吁各大高校、科研院所的科学家们抓紧时间踊跃申报课题立项，领取专项科研经费开展深入合作！
S市科委
2026年1月""",

    "模拟考生5_630字恶意超限粘贴材料卷": """关于2026年公众参与科学项目的发布词
广大市民朋友们：
为营造崇尚科学、尊重科学的浓厚氛围，提升市民科学素养，为知识生产注入新动能，S市科委大力推动“专业科研+公众参与”模式实施。目前网络平台上正持续推进“我的自然百宝箱”“鸟撞调查”“昆虫家谱”“貉口普查”等项目，由不同研究机构发起并每年更新反馈成果。以2019年F大学和市林业总站联合发起的“貉口普查”为例，补充了城市野生动物数据，2023年入选生物多样性优秀实践案例，2025年获政府专项资助。10岁豆豆觉得离科学近了，木匠许师傅吸收知识养分获评优秀参与者，工程师小郭三次当领队改变对科学家高冷印象。团队购置80多台红外相机拍到小灵猫迁移并架设临时浮桥，2023年夏300多位市民周末两晚完成90个小区普查相当于传统团队两个半月工作量，昆虫名录新增物种20%由公众发现。
近些年项目集中在生态学领域，2026年1月我委将联合相关部门高校筹备发布8个新项目，扩展到天文学、心理学等多个领域。该模式打破职业与年龄壁垒，既增加公众对科学的了解，也为科研人员提供启发。项目按“项目发布-公众注册-项目培训-观察记录-数据上传-处理校验-研究结论”推进，欢迎广大市民在“公众参与科学项目”线上平台注册报名，通过在线教程、直播培训、线下讲座学习设备使用和样本记录后积极参与研究！
S市科委
2026年1月""",
}


PBC_SYNTHETIC_ANSWERS = {
    "模拟考生1_抠材料补齐漏点高分卷": """一、助力农民工返岗。上线点对点对接平台，企业填报需求、输入输出地及时对接；成立跨部门协调小组，派专车专列专机护送，实施政府组织企业出资的“包邮”返岗；基层开具健康证明和返岗登记表追溯闭环；启动省际帮扶调剂。
二、稳岗并护航在岗权益。阶段性减免社保医保、缓缴公积金；放宽失业保险返还条件；上门提醒申报扶贫基地领补贴；指导协商调薪、缩工时、轮岗及延期付薪，依法规范裁员，按规定发停工期工资或生活费。
三、拓宽本地与高校毕业生就业。线上线下发岗位、设公益岗稳住贫困户；启动互联网+技能培训并给补贴；教育部等联合推校园网招与国企云招聘，丰富在线面试精准对接。""",

    "模拟考生2_大白话同义改写卷": """1.帮外地务工人员顺利回去上班：建网上对接平台让厂里报人数、两地部门通气；多部门联手派包车专列专机接送，还搞政府牵头企业掏车费的包车行动；村镇开健康证和登记表方便查验追踪；跨省协调调剂用工。
2.帮企业扛压力、护住员工饭碗：暂时少收免收企业社保医保、允许公积金晚交；降低门槛退还部分失业保险费；上门提醒符合条件的厂子申请扶贫基地拿补贴；引导困难厂子和员工商量着调工资、轮流上班或晚点发薪水，盯着别乱裁员，停工期按规矩发工资或生活费。
3.帮滞留老乡和大学毕业生找活干：微信和村里贴单子发岗位、设公益岗帮贫困户就近挣钱；推网上技能培训课并发补贴；教育部等联手招聘网和国企搞线上校招、视频面试帮大学生找工作。""",

    "模拟考生3_错抄企业与专家主体卷": """各级政府采取了以下措施：一、推广信息化办公软件和视频会议系统，规定职工在家远程办公也算正式返岗复工，确保大项目进度不受影响。二、公开承诺不减薪、不裁员、不离弃，号召高管自愿捐出2、3月份一半工资奖励一线环卫工人。三、要求各企业在调整薪资前与员工充分沟通，承诺度过危机后给予相应补偿并签字确认。四、上线用工对接平台，组织专车专列护送农民工返岗。""",

    "模拟考生4_450字超限冗余粘贴卷": """一、畅通返岗路。针对7500万跨省农民工受阻问题，人社部第一时间上线农民工点对点用工对接信息平台，支持企业在线填报需求，实现输入地和输出地对接；会同公安、交通、卫健和国铁集团成立服务协调小组，组织专车、专列、专机运送超80万人；深圳实施政府组织企业出钱的包邮返岗；镇村两级出具健康证明和返岗登记表形成闭环；启动省际帮扶协作调剂。
二、保障在岗权益。国务院决定阶段性减免社保费、医保费和缓缴住房公积金；人社部财政部放宽失业保险返还条件，中小微裁员不超5.5%可返50%，发放423亿元惠及8513万人；人社厅上门提醒申请扶贫基地领30万补助和5000元补贴；指导协商调整薪酬、缩短工时、轮岗轮休、延期支付工资，依法规范裁员，停工期一个周期内按合同付薪、超周期发生活费。
三、促进本地与高校就业。开通公众号和村宣传单发1900个岗位，开辟公益性岗位保贫困户不返贫；启动互联网+职业技能培训计划，选50家平台推100个工种补贴100万人次；教育部联合5家网站推24365校园招聘80万岗位，国投推国聘行动云招聘，办2万场网招。""",
}


def run_q3_experiment(api_key):
    print(">>> 开始实验一：2026年国考地市卷·第3题（发布词，20分，450字）", flush=True)
    q3_path = ROOT / "试卷" / "公务员" / "国考" / "2026年国家公考《申论》题（地市级）" / "第3题.md"
    bundle = load_question_bundle(q3_path)
    constraints = parse_hard_constraints(bundle["题干与要求"])
    spec = call_llm_json(SPEC_PROMPT, {"题干与要求": bundle["题干与要求"]}, api_key=api_key, temperature=0.0)
    print(f"  [题干槽位解析完成] 题型={spec['题型类别']} | 子类={spec['文种子类']} | 号召={spec['是否要求号召动员']}", flush=True)

    card_pack = extract_verified_cards(spec, bundle["材料段落"], api_key=api_key)
    print(f"  [材料3原子拆卡完成] 共 {len(card_pack['功能簇列表'])} 个功能簇，{len(card_pack['卡片'])} 张原文校验卡片", flush=True)

    # 前18份用于建立多机构共识预制表，后6份留作纯盲测机构卷
    train_refs = bundle["参考答案"][:18]
    aligned_18 = align_reference_answers(card_pack["卡片"], train_refs, is_gongwen=True, api_key=api_key)
    print("  [18份机构答案对齐校验完成]", flush=True)

    rubric_multi = compile_quantitative_rubric(
        bundle, spec, constraints, card_pack, aligned_18, mode_label="多机构共识预制(前18份建表)"
    )
    # 对照实验：故意只用第0份（某笔1份答案）编译单答案预制表！
    rubric_single_blind = compile_quantitative_rubric(
        bundle, spec, constraints, card_pack, aligned_18[:1], mode_label="单答案蒙眼预制(仅喂某笔1份答案)"
    )

    # 组装待批改测试卷：6份留出未见机构卷 + 3份典型机构对照卷(某笔、公道公考、饮冰) + 5份千奇百怪模拟考生卷
    test_cases = {}
    for idx in [0, 2, 3]:  # 某笔、公道公考、饮冰(受众错位典型)
        ans = bundle["参考答案"][idx]
        test_cases[f"机构对照_{ans['来源']}"] = ans["正文"]
    for idx in range(18, len(bundle["参考答案"])):
        ans = bundle["参考答案"][idx]
        test_cases[f"盲测机构_{ans['来源']}"] = ans["正文"]
    test_cases.update(Q3_SYNTHETIC_ANSWERS)

    print(f"  [开始双盲批改 {len(test_cases)} 份答卷 (多机构表)]...", flush=True)
    multi_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = {
            name: pool.submit(grade_answer_with_audit, rubric_multi, body, api_key, True)
            for name, body in test_cases.items()
        }
        for name, fut in futs.items():
            multi_results[name] = fut.result()
            aud = multi_results[name]["双盲审计"]
            print(
                f"    - {name}: {multi_results[name]['最终得分']}/20 "
                f"(正序={aud['正序独立检核分']}, 逆序={aud['逆序独立检核分']}, 双盲差={aud['原始双盲分差']})",
                flush=True,
            )

    # 再用“仅喂某笔1份答案”的单答案蒙眼表给同一批5份模拟考生卷+某笔+饮冰+华公余思君算分对比！
    preferred = [
        "机构对照_某笔",
        "机构对照_饮冰",
        "盲测机构_凯叔讲申论",
        "盲测机构_华公余思君",
        "盲测机构_新途径",
    ]
    compare_names = [k for k in preferred if k in test_cases] + list(Q3_SYNTHETIC_ANSWERS.keys())
    print(f"  [开始单答案蒙眼表对照批改 {len(compare_names)} 份代表卷]...", flush=True)
    single_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = {
            name: pool.submit(grade_answer_with_audit, rubric_single_blind, test_cases[name], api_key, False)
            for name in compare_names
        }
        for name, fut in futs.items():
            single_results[name] = fut.result()

    return {
        "多机构预制表": rubric_multi,
        "单答案蒙眼预制表": rubric_single_blind,
        "多机构表批改结果": multi_results,
        "单答案蒙眼表对照结果": {
            k: {
                "多机构表得分": multi_results[k]["最终得分"],
                "单答案蒙眼表得分": single_results[k]["最终得分"],
                "分差": round(single_results[k]["最终得分"] - multi_results[k]["最终得分"], 2),
            }
            for k in compare_names
        },
    }


def run_pbc_experiment(api_key):
    print("\n>>> 开始实验二：2020年中国人民银行·第1题（全库仅某笔1份答案的冷门题，20分，300字）", flush=True)
    pbc_path = ROOT / "试卷" / "事业单位" / "中国人民银行" / "2020年11月7日中国人民银行招聘考试《申论》" / "第1题.md"
    bundle = load_question_bundle(pbc_path)
    constraints = parse_hard_constraints(bundle["题干与要求"])
    spec = call_llm_json(SPEC_PROMPT, {"题干与要求": bundle["题干与要求"]}, api_key=api_key, temperature=0.0)
    card_pack = extract_verified_cards(spec, bundle["材料段落"], api_key=api_key)
    print(f"  [资料2-4原子拆卡完成] 共 {len(card_pack['功能簇列表'])} 个功能簇，{len(card_pack['卡片'])} 张原文校验卡片", flush=True)

    aligned_1 = align_reference_answers(card_pack["卡片"], bundle["参考答案"], is_gongwen=False, api_key=api_key)
    rubric_pbc = compile_quantitative_rubric(
        bundle, spec, constraints, card_pack, aligned_1, mode_label="冷门题单答案纠偏预制(仅某笔1份答案)"
    )

    test_cases = {
        "机构原卷_某笔(含依法处理缩水与漏点)": bundle["参考答案"][0]["正文"],
        **PBC_SYNTHETIC_ANSWERS,
    }
    pbc_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futs = {
            name: pool.submit(grade_answer_with_audit, rubric_pbc, body, api_key, True)
            for name, body in test_cases.items()
        }
        for name, fut in futs.items():
            pbc_results[name] = fut.result()
            aud = pbc_results[name]["双盲审计"]
            print(
                f"    - {name}: {pbc_results[name]['最终得分']}/20 "
                f"(正序={aud['正序独立检核分']}, 逆序={aud['逆序独立检核分']}, 双盲差={aud['原始双盲分差']})",
                flush=True,
            )

    return {
        "单答案纠偏预制表": rubric_pbc,
        "批改结果": pbc_results,
    }


def render_markdown_report(q3_data, pbc_data, elapsed_sec):
    lines = [
        "# 申论量化预制与确定性批改实测报告",
        "",
        f"- 模型：`deepseek-flash`（生成与检核均设 `temperature=0`，禁止模型直接给分）",
        f"- 总耗时：`{elapsed_sec:.1f} 秒`",
        "",
        "---",
        "",
        "## 实验一：2026年国考地市卷 · 第 3 题（反套路公文：发布词，20分，限450字）",
        "",
        "### 1. 题干花哨要求如何变成死规则（分池锁死）",
        "",
    ]
    r3 = q3_data["多机构预制表"]
    p3 = r3["分池配置"]
    lines += [
        f"- **内容分池上限**：`{p3['内容分池上限']} 分`（卡片原始总分 `{p3['卡片弹性总和']} 分`，各功能簇封顶合计 `{p3['簇封顶总和']} 分`，防堆砌同质故事刷分）",
        f"- **文体格式分池（反套路发布词）**：`{p3['文体格式分池']} 分` —— 识别为 `{r3['题干槽位']['文种子类']}`，标题/开场发布宣告(1.0) + 受众称呼或首段直呼市民(1.0)，不因未写落款日期冤扣讲稿式答卷。",
        f"- **“有号召力”客观三要素分池**：`{p3['场景号召与受众分池']} 分` —— ①明确向公众发出报名行动呼吁(1.0)；②客观参与价值支撑联动(1.0，命中>=2张零门槛/收获/成效卡自动得满分)；③受众与身份不串台红线(1.0，把科学家申报课题领经费安到市民头上或只呼吁科学家直接扣分)。",
        f"- **字数与物理截断**：`{r3['字数与扣分死规则']['超字数规则']}`",
        "",
        "### 2. 锁死的功能簇与采分卡片表（前 18 份机构答案 + 材料 3 原文对账生成）",
        "",
        "| 簇编号 | 功能簇名称 | 簇内卡片总分 | 簇封顶分 |",
        "|---|---|---:|---:|",
    ]
    for cr in r3["功能簇封顶表"]:
        lines.append(f"| {cr['簇编号']} | {cr['簇名称']} | {cr['簇内卡片总分']} | **{cr['簇封顶分']}** |")

    lines += [
        "",
        "| 卡号 | 所属簇 | 核心语义 | 面向/惠及对象 | 计分角色 | 参考命中统计 | 满分值 |",
        "|---|---|---|---|---|---|---:|",
    ]
    for sc in r3["采分卡片表"]:
        lines.append(
            f"| {sc['卡号']} | {sc['所属簇']} | {sc['核心语义']} | {sc['面向或惠及对象']} | {sc['计分角色']} | {sc['参考命中统计']} | **{sc['满分值']}** |"
        )

    lines += [
        "",
        "### 3. 14 份答卷（盲测机构卷 + 翻车卷 + 5种奇葩考生卷）批改与双盲一致性结果",
        "",
        "| 答卷名称 | 方格字数 | 内容分 | 格式分 | 号召与受众分 | 超字数扣分 | 最终得分 | 正序/逆序独立分(分差) | 扣分与风险提示 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for name, res in q3_data["多机构表批改结果"].items():
        sub = res["分项得分"]
        aud = res["双盲审计"]
        tips = "；".join(res["扣分与风险提示"]) if res["扣分与风险提示"] else "无"
        lines.append(
            f"| {name} | {res['总格数']} | {sub['内容得分']} | {sub['文体格式得分']} | {sub['场景号召与受众得分']} | -{sub['超字数扣分']} | **{res['最终得分']}** | {aud['正序独立检核分']} / {aud['逆序独立检核分']} ({aud['原始双盲分差']}) | {tips} |"
        )

    lines += [
        "",
        "### 4. 关键验证：假如第 3 题只有“某笔 1 份答案”，单答案蒙眼表打分准不准？",
        "",
        "| 答卷名称 | 18家共识表得分 | 仅喂某笔1份答案蒙眼表得分 | 差值 |",
        "|---|---:|---:|---:|",
    ]
    for name, row in q3_data["单答案蒙眼表对照结果"].items():
        lines.append(f"| {name} | {row['多机构表得分']} | {row['单答案蒙眼表得分']} | {row['分差']:+.1f} |")

    r_pbc = pbc_data["单答案纠偏预制表"]
    lines += [
        "",
        "---",
        "",
        "## 实验二：2020年中国人民银行 · 第 1 题（真·全库仅某笔 1 份答案，20分，限300字）",
        "",
        "### 1. 单答案自动纠偏预制表（自动识破某笔“依法处理”缩水 + 补回漏点）",
        "",
        "| 卡号 | 所属簇 | 核心语义 | 计分角色 | 某笔命中情况 | 满分值 |",
        "|---|---|---|---|---|---:|",
    ]
    for sc in r_pbc["采分卡片表"]:
        lines.append(
            f"| {sc['卡号']} | {sc['所属簇']} | {sc['核心语义']} | {sc['计分角色']} | {sc['参考命中统计']} | **{sc['满分值']}** |"
        )

    lines += [
        "",
        "### 2. 某笔原卷 vs 抠材料高手卷 vs 奇葩卷批改结果",
        "",
        "| 答卷名称 | 方格字数 | 内容得分 | 超字数扣分 | 最终得分 | 正序/逆序独立分(分差) | 扣分与风险提示 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for name, res in pbc_data["批改结果"].items():
        sub = res["分项得分"]
        aud = res["双盲审计"]
        tips = "；".join(res["扣分与风险提示"]) if res["扣分与风险提示"] else "无"
        lines.append(
            f"| {name} | {res['总格数']} | {sub['内容得分']} | -{sub['超字数扣分']} | **{res['最终得分']}** | {aud['正序独立检核分']} / {aud['逆序独立检核分']} ({aud['原始双盲分差']}) | {tips} |"
        )

    return "\n".join(lines) + "\n"


def main():
    api_key = os.environ["DEEPSEEK_API_KEY"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    q3_data = run_q3_experiment(api_key)
    (OUT_DIR / "2026国考地市第3题_预制与实测.json").write_text(
        json.dumps(q3_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    pbc_data = run_pbc_experiment(api_key)
    (OUT_DIR / "2020人民银行第1题_单答案预制与实测.json").write_text(
        json.dumps(pbc_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    elapsed = time.time() - t0
    md = render_markdown_report(q3_data, pbc_data, elapsed)
    report_path = OUT_DIR / "双题量化预制与批改实测报告.md"
    report_path.write_text(md, encoding="utf-8")
    print(f"\n全部实验完成！用时 {elapsed:.1f}s，报告已保存至: {report_path}", flush=True)


if __name__ == "__main__":
    main()
