# 架构说明（ARCHITECTURE.md）

## 一、模块与文件职责

- `shenlun.py`：提供专用浏览器启动入口、手动扫码会话管理与单题答案文本去重导入命令行。
- `collect_paper.py`：按公务员/事业单位目录串行归档整套试卷材料与多机构参考答案，并维护采集断点进度。
- `scoring/engine.py`：基础方格纸计字与物理截断、逐字连续引文防幻觉校验、单答案容量守恒与公文号召力联动实验引擎。
- `scoring/universal_engine.py`：零主观预设通用小题预制、双盲寻证批改、逐分扣分账本生成与原卷微创升格重测闭环引擎（支持归纳概括、分类、观点表态、问题反推对策及各类反套路公文）。
- `scoring/essay_engine.py`：预制作文立意要求，核查答卷证据，合成档位得分，并诊断回忆提纲。
- `scoring/paper_pipeline.py`：加载题目、材料和参考，制作并缓存带题型的细则，完整卷汇总总分。
- `scoring/rubric_grading.py`：定义三类题的制作与阅卷提示，调用模型按细则批改并校验结果。
- `scoring/__main__.py`：单题制作和复用已保存细则批改的命令行入口。
- `rubric_workshop/`：独立执行拆题、材料预处理、参考答案核对、分值与判据制作、试评和锁定，并支持读取逐题细则判卷。
- `scoring/run_batch_calibration.py`：载入同卷回忆答案并生成逐题证据与已知总分的对照报告。
- `scoring/rubric_cache.py`：按建表输入和版本存放逐题细则，用文件锁避免同版本重复预制，原子写入完成产物。
- `scoring/run_corpus_calibration.py`：按年份与卷别匹配答卷与试卷，批跑多卷并保留双轮、首轮和失败记录。
- `scoring/v0_frozen/`：原样冻结的评分源文件，供评测基线调用；摘要记录在 `eval/v0_frozen.sha256`。
- `eval/`：锁定按整卷划分的数据和对照底稿，为每个变体独立核验，运行恒猜、直接提示词与冻结引擎并汇总指标。
- `scoring/run_experiments.py` / `scoring/run_4_guokao_test.py`：针对多答案发布词、单答案冷门题及4道反套路国考真题的端到端实测脚本。
- `test_navigation.py` / `test_catalog.py`：本地模拟网页校验浏览器启动限制检测与试卷目录采集逻辑。
- `test_scoring_math.py`：离线校验方格纸计字、物理边界截断、引文逐字防幻觉匹配、案例簇封顶与公文号召力客观联动算分。
- `预制与实测/`：存放题目锁定预制表 JSON 与双盲批改对照实测报告。

- `scoring/class_a_pipeline.py` 提供热门题实验入口，支持为制标与标杆阶段选择 Responses 模型。
- `scoring/reference_workflow.py` 执行参考稿筛选、共识对齐、细则冻结、标杆生成和匿名批改。制作缓存按模型与输入区分，阅卷以冻结细则为准。标杆蓝图验真统领短引，成稿按普通小题连续排版和公文必要换行分别核算。

## 二、模块调用关系

1. **题库归档链路**：`run.cmd` -> `shenlun.py` -> `collect_paper.py` -> 生成 `试卷/<考试大类>/<地区>/<卷名>/材料*.md` 与 `第*题.md`。
2. **制作每题细则**：`prefab_paper_question` 载入题干、材料、参考答案，一次模型调用按顺序判题型并制作细则。缓存键绑定模型、材料、题干、参考答案和制作提示词。
3. **批改单题或整卷**：`grade_rubric_answer` 把细则、题干、材料和答卷交给模型。程序验证答卷引文、逐项分值或作文档位；小题汇总模型给出的分项，完整卷汇总逐题得分。整卷入口支持旧报告细则复测。

## 三、关键设计决定及原因

1. 第一类按要点给分；第二类固定 75% 要点和 25% 逻辑格式；第三类先定档，再给档内分。
2. 机构答案仅用于制作细则。保存的单题细则包包含题干与材料，方便批改复用。
3. 小题按题干格数上限截取可见内容。作文交给阅卷员按整篇答卷和细则处理篇幅要求。

- `rubric_workshop/blind_batch.py` 读取五年题库，分阶段执行无参考制标、独立复核及冻结后的署名盲评。
- `rubric_workshop/blind_report.py` 汇总逐题分数和低分诊断，输出可查阅的结果索引。

- `rubric_workshop/semantic_rubric.py` 从题目与材料生成逐段事实、方面草案，复核可省略细节和最低充分表达，冻结后交给 `blind_batch.evaluate_aspects` 按原句评分。

- `rubric_workshop/check_independent_tasks.py` 从相同题源细则检验内容是否应独立计分，保留预算，冻结后用旧卷与留出卷核对逐项变化。
