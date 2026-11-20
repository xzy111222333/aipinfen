# 逐题评分标准与细则

从仓库根目录运行。制作沿用项目配置的模型和题库读取方式。

```powershell
$env:PYTHONUTF8 = '1'
python -m rubric_workshop make
python -m rubric_workshop make --question 2 4
python -m rubric_workshop grade 'rubric_workshop/2026国考地市/第2题.json' '答卷.txt'
python -m rubric_workshop grade 'rubric_workshop/2026国考地市/人工核对/第2题.json' '答卷.txt'
python -m rubric_workshop.verify_paper --directory 'rubric_workshop/2026国考地市/人工核对' 1 2 3 4 5
python -m unittest rubric_workshop.test_workshop
```

制作依次执行拆题判型、独立读材料、逐份核对参考答案、分配分值、写出给分边界、试评修订。第 1 类按要点给分，第 2 类固定内容 75%、逻辑格式 25%，第 3 类按四类八档给分。题库标注的考生回忆与带分答案从制作输入排除。

Markdown 是阅卷人可阅读的细则，JSON 同时记录题干材料、参考来源、材料依据、规则和试评记录。批改读取已锁定的 JSON，使用题干材料和细则逐项判分。程序核对原文引证、固定分值和档位。试评是产品细则的内部一致性检查，具体分值是拟定尺度，绝对分还需真实样卷校准。

支持 `--paper` 和 `--output` 指定其他题库目录与输出位置。试评未通过时，JSON 标注待修订并保留差异。CLI 返回非零退出码。

首卷的《人工核对稿》和对应 JSON 来自人工按题干与材料拟定。自动制作的结果逐题单独保存。用相同的一组独立对照可以分别核查这两套细则。

制作时逐份匿名核对参考答案。定稿逐条记录候选归入哪个评分项，或为何只作例证、为何排除。首卷输入数量和实际分歧见《参考答案核对》。

## 五年小题无机构答案批次

从仓库根目录依次运行 `python -m rubric_workshop.blind_batch make`、`python -m rubric_workshop.blind_batch verify`、`python -m rubric_workshop.blind_batch grade` 和 `python -m rubric_workshop.blind_report`。批次覆盖2022至2026年十五套卷的第一、二类题及两道短评，题型取自此前只看题干和要求的逐题分类。可以用 `--key 2025执法-1` 指定题位。

初制只发送题目、作答要求、指定材料和既定类别预算。段号、证据引文、各类分值合计及满分边界由程序核对；独立复核对照题干材料提出实质问题和边界样例，修订或标记待复核。冻结后才提取署名白鹭的答卷试评，排除明确标为第三方改进的版本，保存细则文件摘要和逐项答卷证据。低于满分60%的版本逐一审查，诊断不回写细则。各版白鹭答案均无官方得分。试评不充当绝对分校准。

每轮修改制标或判分前先读[小题制标原则与回测](小题制标原则与回测.md)。从根目录运行 `python -m rubric_workshop.joint_replay --output rubric_workshop/五年小题无参考/联合验收` 可继续该轮验收；变更规则时换用新目录，保留前一轮冻结细则和答卷结果。对照检查同时保存各项分差，须复核是否误扣无关项。
