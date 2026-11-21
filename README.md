# 申论真题归档与量化批改系统

## 一、功能简介

2026-11-21 重采版通过 `run.cmd browser --stealth --archive-dir 试卷_含赞踩_2026-11-21` 启动。在专用浏览器中复用登录会话后，终端输入 `batch` 开始全量采集。新目录独立保存归档和断点，每份答案额外记录点赞、点踩与采集时间。读取到空白或无法解析的赞踩计数时会报错，以便核对网页。目录翻页等待 3 秒，普通操作等待 1 秒，答案区仍连续检查两次稳定状态。

本项目解决申论备考的两个核心痛点：
1. **多机构答案汇集归档**：申论从不公布官方答案，单一机构答案往往存在遗漏或套话。本工具将历年公务员、事业单位真题的材料与各家名师参考答案统一归档为本地结构化文本（已归档 1764 套试卷、6066 道题目）。
2. **AI 逐题制作与批改**：制作时根据题目、材料及机构参考答案判定三类题型并保存细则。批改时用同一份细则和本题资料评阅考生答案，返回得分及依据。

## 二、技术架构

- **语言与运行环境**：Python 3.11、Google Chrome（采集模块使用 Playwright，轻量反检测编译可选 JDK 21）。
- **题库存储层**：`试卷/` 目录下按 `考试大类/地区/试卷名/` 存放 UTF-8 格式的 `材料*.md` 与 `第*题.md`。
- **评分层**：`scoring/paper_pipeline.py` 负责制作细则和整卷汇总，`scoring/rubric_grading.py` 按锁定细则评分并校验分值和证据。

## 三、本地运行与部署命令

### 1. 环境初始化

在项目根目录执行：

```powershell
.\setup.cmd
```

（注意：严禁访问 `https://help.srnz.net/` 下的一切网页，该子域名存在广告与异常风险；其余题库页面正常。）

### 2. 题库浏览与归档命令

- 打开专用浏览器扫码登录：`.\run.cmd browser --stealth`
- 去重导入单题多机构答案文本：
  ```powershell
  .\run.cmd import-answers answers.txt paper-875-q1.json --paper-id 875 --question 1
  ```
- 将手动或批量保存的材料与题目归档入库：
  ```powershell
  .\run.cmd archive-paper .\924文本 --category 公务员 --region 国考 --paper-id 924 --title '2026年国家公考申论题（副省级）' --materials 4 --questions 5 --refs '1=1'
  ```

### 3. 制作细则并批改

独立制作流程和 2026 年国考地市卷的逐题细则见 [rubric_workshop](rubric_workshop/README.md)。从项目根目录运行 `python -m rubric_workshop make`，可按题生成可阅读的 Markdown 与可复用的 JSON。

单题制作和批改在项目根目录运行。参数依次为题目文件及输出细则、已保存的细则及 UTF-8 答卷。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scoring prepare '题目文件路径' 'rubric.json'
.\.venv\Scripts\python.exe -X utf8 -m scoring grade 'rubric.json' 'answer.txt'
```

整卷校验可以传入同一试卷的多份回忆答卷。下面用 2022 年副省级完整答卷跑一次，生成逐题得分、证据账本和总分分差。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scoring.run_batch_calibration `
  '试卷/公务员/国考/2022年国家公考《申论》题（副省级）' `
  '国考申论高分答案_MD整合包/完整答案/78.5分-2022年副省级未署名.md' `
  --output '预制与实测/整卷校验_2022副省.json'
```

默认使用题目、材料和全部可用参考答案制作。可通过 `--ref-limit 8` 限制参考数量。加 `--rubric-from` 并指定已有报告，可固定同一份评分细则重测。残卷报告只汇总已记录内容的诊断，整卷总分留空。

### 4. 对照评测

当前批改读取工作区 `.env` 或进程环境中的 `SHENLUN_CHYYDS_API_KEY`，使用 `chyyds.com` 的 `deepseek-v4.1-flash`。未配置该项时使用 `DEEPSEEK_API_KEY` 直连。正式基线固定使用直连，细则缓存按模型和线路分别保存。

评测集按整套试卷划分。生成变体使用 `python -m eval.generate_variants --split dev`，先独立核验变体，再执行基线。正式开发集命令如下。同一运行名接续已有断点。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m eval.run_eval --split dev --engine B0 B1 B2 --run-name 2026-11-18_baselines_dev
```

三种基线分别为恒猜、直接提示词和冻结版引擎。`eval/reports/` 存放指标和成绩单，`eval/runs/` 保留逐题判分、证据与用量。留出集按任务书在里程碑运行。人工两两比较使用 `eval/human_pairs/pairs_todo.md`，填写后用 `python -m eval.human_pairs --import-labels` 导入。

命令默认复用 `预制与实测/评分细则缓存/` 中的逐题细则。题干、材料、所选参考答案或预制提示词变化时，会生成新版细则。同版本并发请求共用一次预制，评分仍逐份核查答卷。可用 `--cache-dir` 指定默认目录以外的缓存位置。

批跑多套试卷及其回忆答卷时，可按年份和卷别自动匹配原卷。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scoring.run_corpus_calibration `
  '试卷/公务员/国考' `
  '国考申论答案_2020-2026_MD整合包/完整答案' `
  '国考申论高分答案_MD整合包/完整答案' `
  --output '预制与实测/真实完整卷_同表双轮与首轮.json'
```

报告同时保留当前双轮结果和同一轮实验中的正序首轮分。首轮分可用于估计去掉复核的影响，实际批改默认仍走双轮。未完成的样本记录原因并使任务返回失败状态。

## 四、测试命令

- 运行量化计字、物理截断与算分公式单元测试（无需联网）：
  ```powershell
  .\.venv\Scripts\python.exe -m unittest test_scoring_math -v
  ```
- 运行浏览器导航与目录归档本地模拟测试：
  ```powershell
  .\.venv\Scripts\python.exe -m unittest test_navigation test_catalog
  ```

## 五、搜索记录

针对“如何消除大模型主观评分漂移并构建可复现申论量表”，已在 `skills.sh` 与 GitHub 检索主流方案并提炼结论：
- **skills.sh 参考方案**：检索 `judge-rubric`（`tyler-r-kendrick/copilot-auto-training`）、`eval-rubric-design`（`dvy1987/agent-loom`）、`criterion-referenced-rubric-generator`（`garethmanning/education-agent-skills`）。核心结论：评分标准必须在批改前锁定（Locked Rubric），禁止用评价性形容词（如“号召力强、内容好”），必须全部改写为可观测文本事实与二元门禁（Pass/Fail Gates），且制定标准与执行核对必须分离。
- **GitHub 参考方案**：检索 `LabRAI/Rulers`（锁定清单 Bundle + 原文连续子串逐字校验 + 确定性分数映射）与 `ye0619/Shenlun-SKILL`、`jeneyzhang016-cyber/guokao-shenlun-grader`（先由多份参考答案与材料建立采分点共识表，再按证据位置—考生原句核对）。
- **落地取舍**：吸收上述“锁定量表 + 逐字引文防幻觉”机制，并额外补齐现有开源方案缺失的三块硬算法——**功能簇封顶**（解决多案例选答）、**方格纸物理截断**（防超字数作弊）、**单答案字数容量守恒与反缩水还原**（解决冷门题只有某笔 1 份中庸答案的困境）。

## 六、已完成与待办

- **已完成**：
  - 归档 1764 套公务员与事业单位试卷（共 6066 道题）。
  - 完成 `scoring/universal_engine.py` 零主观预设通用小题引擎：自动从参考答案实证率生长外壳格式门禁、支持问题自动镜像生成反推对策卡、生成精确到分的扣分账本点评、并在用户原卷基础上做微创升格与重测验分闭环。
  - 累计实测跑通 6 道跨度极大的真题（2026地市第3题发布词、2026地市第4题原因+反推建议、2026副省第2题传统工艺看法表态、2026副省第3题指定标题短评、2025执法第3题夜市巡查工作指南、2020人民银行第1题单答案冷门题）。
- **待办**：
  - 校验大作文立意、论证、篇幅及材料复用指标在更多样本上的表现。
  - 按需批量预制题库中常用国省考整卷并封装本地批改界面。

近五年国考小题支持仅使用题干、要求和给定材料制作细则，按题位冻结后可批量盲评署名白鹭的答案。[命令与批次范围](rubric_workshop/README.md) · [逐题结果索引](rubric_workshop/五年小题无参考/结果索引.md)

# aipinfen
