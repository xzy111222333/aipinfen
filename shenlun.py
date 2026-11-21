import argparse
import json
import os
import re
import subprocess
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory
from datetime import datetime


CARD_BOUNDARY = re.compile(
    r"(?m)^[^\S\n]*隐藏[^\S\n]*(?:\n[^\S\n]*|"
    r"\[image\]\(https://srnz\.net/img/answer-pin\.[^\s)]+\.svg\)[^\S\n]*)?"
    r"置顶[^\S\n]*$"
)
ARCHIVE = Path(__file__).resolve().parent / "试卷"
PAPER_LIST = "https://srnz.net/community/studyRoom/paper"
ACCESS_NOTICE = re.compile(r"请完成(?:人机|安全)验证|访问过于频繁|访问被拒绝|登录已过期")


class NavigationStopped(RuntimeError):
    pass


class TemporaryNavigationFailure(NavigationStopped):
    pass


def is_network_failure(error):
    return any(code in str(error) for code in (
        "ERR_INTERNET_DISCONNECTED", "ERR_NETWORK_CHANGED", "ERR_CONNECTION_RESET",
        "ERR_CONNECTION_CLOSED", "ERR_CONNECTION_TIMED_OUT", "ERR_NAME_NOT_RESOLVED",
    ))


def open_page(page, url):
    from playwright.sync_api import Error, TimeoutError

    try:
        response = page.goto(url, wait_until="domcontentloaded", timeout=15_000)
    except Error as error:
        if isinstance(error, TimeoutError) or is_network_failure(error):
            raise TemporaryNavigationFailure(f"页面网络请求失败：{url}") from error
        raise NavigationStopped(f"页面加载失败，已停止。\n{error}") from error
    if response is not None and response.status >= 400:
        raise NavigationStopped(f"页面返回 HTTP {response.status}，已停止。")

    notice = page.get_by_role("alert").filter(has_text=ACCESS_NOTICE, visible=True).first
    if notice.is_visible():
        raise NavigationStopped("页面提示需要验证、重新登录或限制访问，已停止。")


def read_answers(source):
    blocks = CARD_BOUNDARY.split(source.read_text(encoding="utf-8-sig").strip())
    answers = []
    seen = set()
    for index, block in enumerate(blocks, start=1):
        if not block.strip() and index == len(blocks) and answers:
            break
        author, _, body = block.strip().partition("\n")
        author = author.strip()
        body = body.strip()
        if not author or not body:
            raise ValueError(f"第 {index} 条缺少署名或正文，请核对复制的文本")
        key = (author, body)
        if key not in seen:
            seen.add(key)
            answers.append({"author": author, "text": body})

    return answers


def import_answers(source, output, paper_id, question):
    answers = read_answers(source)
    record = {"paper_id": paper_id, "question": question, "answers": answers}
    with output.open("x", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"已导出 {len(answers)} 条答案：{output}")


def read_required(path):
    content = path.read_text(encoding="utf-8-sig").strip()
    if not content:
        raise ValueError(f"文件为空：{path}")
    return content


def archive_paper(args):
    if args.paper_id < 1 or args.materials < 1 or args.questions < 1:
        raise ValueError("试卷 ID、材料数和题数必须是正整数。")

    region = args.region.strip()
    if (
        not region or region.endswith(".") or PureWindowsPath(region).is_reserved()
        or re.search(r'[<>:"/\\|?*\x00-\x1f]', region)
    ):
        raise ValueError("地区必须是有效的文件夹名称。")

    references = {}
    for entry in args.refs:
        if not re.fullmatch(r"[0-9]+=[0-9]+(?:,[0-9]+)*", entry):
            raise ValueError("材料对应关系格式为 --refs 1=1,2。")
        question, numbers = entry.split("=")
        question = int(question)
        numbers = list(dict.fromkeys(map(int, numbers.split(","))))
        if question in references or not 1 <= question <= args.questions or not all(
            1 <= number <= args.materials for number in numbers
        ):
            raise ValueError(f"材料对应关系重复或编号越界：{entry}")
        references[question] = numbers

    materials = [
        read_required(args.source / f"材料{number}.txt")
        for number in range(1, args.materials + 1)
    ]
    questions = [
        (read_required(args.source / f"第{number}题.txt"),
         read_answers(args.source / f"第{number}题答案.txt"))
        for number in range(1, args.questions + 1)
    ]
    write_paper(args.category, region, args.paper_id, args.title, materials, questions, references)


def paper_folder_name(title):
    name = title.translate(str.maketrans('<>:"/\\|?*', '＜＞：＂／＼｜？＊'))
    name = re.sub(r"[\x00-\x1f]+", " ", name).strip().rstrip(". ")
    if not name or PureWindowsPath(name).is_reserved():
        raise ValueError(f"试卷标题无法用作目录名：{title}")
    return name


def archive_has_id(folder, paper_id):
    first_question = folder / "第1题.md"
    if not first_question.is_file():
        first_question = folder / "试卷说明.md"
    if not first_question.is_file():
        return False
    with first_question.open(encoding="utf-8") as file:
        header = [next(file, "").strip() for _ in range(3)]
    return header[2] == f"来源：{PAPER_LIST}/detail/{paper_id}"


def already_archived(category, region, paper_id):
    parent = ARCHIVE / category / region
    return any(archive_has_id(folder, paper_id) for folder in parent.iterdir()) if parent.is_dir() else False


def write_paper(category, region, paper_id, title, materials, questions, references):
    files = {}
    if not questions:
        files["试卷说明.md"] = (
            f"# {title}\n\n来源：{PAPER_LIST}/detail/{paper_id}\n\n"
            "采集时网页展示了材料，题目区为空，并显示「暂无参考答案」。\n"
        )
    for number, material in enumerate(materials, 1):
        files[f"材料{number}.md"] = f"# 材料 {number}\n\n{material}\n"

    counts = []
    captured_at = datetime.now().astimezone().isoformat(timespec="seconds")
    for number, (question, answers) in enumerate(questions, 1):
        counts.append(len(answers))
        material_numbers = references.get(number, range(1, len(materials) + 1))
        material_links = "、".join(f"[材料 {item}](材料{item}.md)" for item in material_numbers)
        label = "关联材料" if number in references else "全卷材料"
        sections = [
            f"# {title} · 第 {number} 题",
            f"来源：{PAPER_LIST}/detail/{paper_id}",
            f"## {label}",
            material_links,
            "## 题目",
            question,
            f"## 答案（{len(answers)} 份）",
        ]
        for answer in answers:
            sections.append(f"### {answer['author']}")
            if "likes" in answer and "dislikes" in answer:
                sections.append(
                    f"点赞 {answer['likes']} · 点踩 {answer['dislikes']} · 采集时间 {captured_at}"
                )
            sections.append(answer["text"])
        files[f"第{number}题.md"] = "\n\n".join(sections) + "\n"

    parent = ARCHIVE / category / region
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / paper_folder_name(title)
    if output.exists():
        if archive_has_id(output, paper_id):
            raise FileExistsError(f"试卷目录已存在：{output}")
        output = parent / f"{output.name}（{paper_id}）"
        if output.exists():
            raise FileExistsError(f"同名试卷目录已存在：{output}")
    with TemporaryDirectory(prefix=".归档-", dir=parent) as staging:
        paper = Path(staging) / "paper"
        paper.mkdir()
        for name, content in files.items():
            (paper / name).write_text(content, encoding="utf-8")
        paper.rename(output)
    print(f"已归档 {len(materials)} 份材料、{len(questions)} 道题，各题答案数：{counts}\n{output}")


def open_browser(check_only, stealth=False, paper_id=None):
    if check_only:
        with TemporaryDirectory(prefix="shenlun-browser-") as profile:
            run_browser(profile, check_only, stealth)
        return

    profile = Path(os.environ["LOCALAPPDATA"]) / "Shenlun" / "browser"
    run_browser(profile, check_only, stealth, paper_id)


def load_stealth_script():
    root = Path(__file__).resolve().parent
    sources = root / "playwright-stealth-pool-main" / "src" / "main" / "java"
    with TemporaryDirectory(prefix="shenlun-java-") as classes:
        subprocess.run(
            ["javac", "-encoding", "UTF-8", "-sourcepath", str(sources),
             "-d", classes, str(root / "PrintStealthScript.java")],
            check=True,
        )
        result = subprocess.run(
            ["java", "-Dfile.encoding=UTF-8", "-cp", classes, "PrintStealthScript"],
            check=True,
            stdout=subprocess.PIPE,
            encoding="utf-8",
        )
    return result.stdout


def check_browser_script(page):
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.route("**/*", lambda route: route.fulfill(
        content_type="text/html",
        body="""<script>
            window.browserAtLoad = {
                webdriver: typeof navigator.webdriver,
                languages: Array.from(navigator.languages),
                platform: navigator.platform
            };
        </script>""",
    ))
    native_properties = None
    for url, expected in (
        ("https://baseline-test.invalid/community/studyRoom/paper", "boolean"),
        ("https://local-test.invalid/login", "boolean"),
        ("https://local-test.invalid/community/studyRoom/paper-other", "boolean"),
        ("https://local-test.invalid/community/studyRoom/paper", "undefined"),
        ("https://local-test.invalid/community/studyRoom/paper/detail/924", "undefined"),
    ):
        open_page(page, url)
        if errors:
            raise NavigationStopped(f"本地测试页面脚本报错：{errors[0]}")
        actual = page.evaluate("window.browserAtLoad")
        if actual is None or actual.pop("webdriver") != expected:
            raise NavigationStopped(f"本地测试失败：{url} 的浏览器属性异常：{actual}")
        if native_properties is None:
            native_properties = actual
        elif actual != native_properties:
            raise NavigationStopped(f"本地测试失败：{url} 的语言或平台与原生浏览器不一致。")
        print(f"{url} 页面加载时的 webdriver 类型：{expected}")


def run_browser(profile, check_only, stealth, paper_id=None):
    from playwright.sync_api import sync_playwright

    try:
        script = load_stealth_script() if stealth else None
    except FileNotFoundError as error:
        raise NavigationStopped("无法启动 Java 工具，请确认 JDK 21 的 java 和 javac 在 PATH 中。") from error
    except subprocess.CalledProcessError as error:
        raise NavigationStopped(f"反检测脚本加载失败，{error.cmd[0]} 退出码为 {error.returncode}。") from error
    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            profile, channel="chrome", headless=False, chromium_sandbox=True
        )
        try:
            if script:
                origin = "https://local-test.invalid" if check_only else "https://srnz.net"
                context.add_init_script(
                    rf"""if (window === window.top &&
                        location.origin === {json.dumps(origin)} &&
                        /^\/community\/studyRoom\/paper(?:\/detail\/\d+)?\/?$/.test(location.pathname)) {{
                        {script}
                    }}"""
                )
            page = context.pages[0] if context.pages else context.new_page()
            if check_only and stealth:
                check_browser_script(page)
            else:
                url = f"{PAPER_LIST}/detail/{paper_id}" if paper_id else PAPER_LIST
                open_page(page, "about:blank" if check_only else url)
            if check_only:
                print("Chrome 启动检查通过。")
            else:
                while command := input("输入 inspect、collect、check-catalog、sample 或 batch，按回车关闭浏览器：").strip():
                    if command in {"batch", "check-catalog", "sample"}:
                        from collect_paper import collect_catalog
                        from playwright.sync_api import Error, TimeoutError

                        try:
                            delays = iter((30, 120, 300) if command == "batch" else ())
                            while True:
                                try:
                                    collect_catalog(
                                        page, open_page, PAPER_LIST, write_paper, NavigationStopped,
                                        check_only=command == "check-catalog",
                                        sample=command == "sample",
                                        archived=already_archived if command == "batch" else None,
                                        progress=ARCHIVE / "采集进度.json" if command == "batch" else None,
                                        limit=None,
                                    )
                                    break
                                except (TemporaryNavigationFailure, Error) as error:
                                    delay = next(delays, None)
                                    if delay is None or page.is_closed() or not (
                                        isinstance(error, (TemporaryNavigationFailure, TimeoutError))
                                        or is_network_failure(error)
                                    ):
                                        raise
                                    print(f"网络异常，{delay} 秒后从已保存的位置恢复：{error}", flush=True)
                                    page.wait_for_timeout(delay * 1000)
                        except (NavigationStopped, Error, AssertionError, OSError, ValueError) as error:
                            print(f"目录操作停止：{error}", flush=True)
                        continue
                    if command == "catalog":
                        open_page(page, PAPER_LIST)
                        page.get_by_text("公务员考试", exact=True).wait_for(timeout=10000)
                        print(page.locator("body").evaluate(r"""body =>
                            [...body.querySelectorAll('*')]
                            .filter(e => e.getClientRects().length &&
                                (/^20\d{2}年[^\n]+$/.test(e.innerText?.trim()) ||
                                 /^(所有省份|所有地区|国考)$/.test(e.innerText?.trim())))
                            .slice(0, 22).map(e => [e.tagName, e.className,
                                e.parentElement.className, e.innerText.trim().slice(0, 65)])
                        """))
                        continue
                    if command not in {"inspect", "collect", "inspect-votes"}:
                        print("未知命令。")
                        continue
                    current = next((tab for tab in context.pages if tab.url.startswith(f"{PAPER_LIST}/detail/")), None)
                    if current is None:
                        print("当前没有打开试卷详情。")
                        continue
                    if command == "inspect-votes":
                        print(current.locator(".user-add-answer-item .answer-header").first.evaluate(
                            """node => [...node.querySelectorAll('.action-item button')].map(
                                button => ({classes: button.className, text: button.innerText.trim()})
                            )"""
                        ))
                        continue
                    if command == "collect":
                        from collect_paper import collect_paper

                        try:
                            collect_paper(current, write_paper, NavigationStopped)
                        except (NavigationStopped, OSError, ValueError) as error:
                            print(error)
                        continue
                    labels = current.locator("body").evaluate(r"""body => {
                        const labels = /^(?:第\d+题|材料\d+|隐藏|置顶)$/;
                        return Array.from(body.querySelectorAll('*'))
                            .filter(node => node.getClientRects().length &&
                                labels.test((node.innerText || '').replace(/\s/g, '')) &&
                                !Array.from(node.children).some(child =>
                                    labels.test((child.innerText || '').replace(/\s/g, ''))))
                            .map(node => ({
                                label: node.innerText.trim(),
                                selector: [node.parentElement, node].map(element =>
                                    element.tagName.toLowerCase() +
                                    Array.from(element.classList, name => '.' + CSS.escape(name)).join('')
                                ).join(' > ')
                            }));
                    }""")
                    headings = current.get_by_role("heading").all_inner_texts()
                    print(json.dumps({"url": current.url, "headings": headings, "labels": labels}, ensure_ascii=False))
        finally:
            context.close()


def main():
    global ARCHIVE
    parser = argparse.ArgumentParser(description="申论答案本地导入和浏览器入口")
    commands = parser.add_subparsers(dest="command", required=True)
    browser = commands.add_parser("browser", help="打开专用浏览器窗口")
    browser.add_argument("--blank", action="store_true", help="打开空白页验证环境")
    browser.add_argument("--stealth", action="store_true", help="使用 ZIP 项目的轻量反检测脚本")
    browser.add_argument("--paper-id", type=int, help="直接打开指定试卷")
    browser.add_argument("--archive-dir", type=Path, help="本轮归档目录")
    importer = commands.add_parser("import-answers", help="从本地文本导入一道题的答案")
    importer.add_argument("source", type=Path)
    importer.add_argument("output", type=Path)
    importer.add_argument("--paper-id", type=int, required=True)
    importer.add_argument("--question", type=int, required=True)
    archive = commands.add_parser("archive-paper", help="将本地材料、题目和答案整理成整卷")
    archive.add_argument("source", type=Path, help="存放复制文本的目录")
    archive.add_argument("--category", required=True, choices=("公务员", "事业单位"))
    archive.add_argument("--region", required=True, help="网站上的地区或卷类，例如国考")
    archive.add_argument("--refs", action="append", default=[], help="题目关联材料，例如 1=1,2")
    archive.add_argument("--paper-id", type=int, required=True)
    archive.add_argument("--title", required=True)
    archive.add_argument("--materials", type=int, required=True)
    archive.add_argument("--questions", type=int, required=True)
    args = parser.parse_args()
    if args.command == "browser":
        if args.archive_dir:
            ARCHIVE = args.archive_dir.resolve()
            ARCHIVE.mkdir(parents=True, exist_ok=True)
        try:
            open_browser(args.blank, args.stealth, args.paper_id)
        except NavigationStopped as error:
            parser.exit(1, f"{error}\n")
    elif args.command == "archive-paper":
        try:
            archive_paper(args)
        except (OSError, ValueError) as error:
            parser.exit(1, f"{error}\n")
    else:
        if args.paper_id < 1 or args.question < 1:
            parser.error("试卷 ID 和题号必须是正整数")
        import_answers(args.source, args.output, args.paper_id, args.question)


if __name__ == "__main__":
    main()
