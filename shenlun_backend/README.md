# 申论小题批改后端

整个目录可独立移到小程序的 Python 后端。运行环境为 Python 3.11，依赖均来自标准库。小程序前端通过已有业务接口提交题目 ID 和答卷，模型请求、密钥、冻结细则和批改缓存都放在服务端。

## 两个入口

| 入口 | 何时调用 | 产物 |
|---|---|---|
| `prepare_question` | 后台新增题目或制作新版本时 | 冻结细则、标杆原稿、盲评明细和全流程记录 |
| `grade_submission` | 考生交卷时 | 总分、内容分、逻辑格式分、逐项证据与反馈、细则指纹 |

在本目录运行命令，或把本目录加入后端的 Python 模块搜索路径。后台配置 `SHENLUN_MDKJ_RESPONSES_API_KEY` 供 GPT 使用，`SHENLUN_MDKJ_API_KEY` 供 DeepSeek 使用。代码沿用已验证的 mdkj.lol 中转接口，分别走 `/v1/responses` 和 `/v1/messages`。部署环境需能访问该服务。真实密钥由部署环境注入。

## 每题预制一次

```sh
python -m scoring prepare examples/night_market/第3题.md data/night-market/v1 --holdout 白鹭 --sheet 25 20
```

题目文件的「关联材料」用相对路径链接材料文件，「题目」含要求、满分和字限，「答案」保留机构来源、赞踩行和正文。示例目录给出了可直接运行的一整题。`--holdout` 支持多次传入，关联来源及相似留出稿在归纳前剔除。省略该参数时只复评标杆。此流程面向有多份参考的小题，筛选后至少需要三份可用稿。

1. GPT‑6.2‑sol 提取机构答卷正文，去除讲解文字，结合题干和原材料归纳各机构答出的实质意思。
2. 制标按好评率的威尔逊下界乘 `ln(1+点赞数)` 衡量机构共识，冻结内容项、分值、部分得分条件及逻辑格式项。
3. 写标杆时单独按「威尔逊下界 × 点赞数平方根」计算并归一化权重。默认向写作环节提供最多五份入选正文；`--writing-count 3` 可恢复常规三份配置。每份附赞、踩和写作权重。GPT 依次生成提纲、两份候选并选优，再复核定稿。
4. 有答题区尺寸时传入每行格数和总行数。缺少尺寸的公文按每行 25 格和字限模拟，并记录这个假设。计格按原稿实际缩进和换行执行。
5. DeepSeek‑v4.1‑flash 对标杆和留出稿按同一冻结细则评分。`全流程结果.json` 成功生成后，业务后端再发布这个题目版本。每次新版本使用独立目录，同一版本由一个后台任务负责生成；中断后可在原目录重试并复用有效阶段缓存。

## 小程序交卷

以下代码在业务后端执行。由服务端根据题目 ID 查找已发布的细则路径；评分函数会同步等待模型返回，耗时请求可交给已有任务队列执行。

```python
from scoring import grade_submission

def score_night_market(answer: str):
    return grade_submission(
        "examples/night_market/冻结细则.json",
        answer,
        "data/grading/night-market/v1",
    )
```

命令行也支持读取 UTF-8 答卷并输出 JSON：

```sh
python -m scoring grade examples/night_market/冻结细则.json answer.txt --cache data/grading/night-market/v1 --output data/submission.json
```

`内容判定` 含每项满分、得分、答卷证据原句和反馈。相同版本下的答卷可复用缓存。缓存与预制目录须可持久写入，并仅供服务端访问。采用 Node 等其他后端时，可从已有任务执行器以参数数组调用以上 CLI，按退出码判断成功后读输出文件；每份提交使用独立输出文件。

## 接入核对

`examples/night_market/标杆答案.json` 是最后认可的五份带权参考原稿，424 个有效格，模拟排版 494 格、20 行。`冻结细则.json` 是当时实际使用的评分包，`复评分.json` 是已记录的 DeepSeek 逐项评分。新调用生成的措辞可能不同。使用 `python -m unittest -q test_portable` 可离线核对随包题目、参考权重、排版和细则指纹。
