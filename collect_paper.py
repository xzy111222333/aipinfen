import re
import random
import json
from tempfile import NamedTemporaryFile
from pathlib import Path
from datetime import datetime

from playwright.sync_api import Error, TimeoutError, expect


ACTION_DELAY_MS = 1000
PAGE_DELAY_MS = 3000
ANSWER_SETTLE_MS = 800


def read_answer_cards(cards):
    records = cards.evaluate_all("""cards => cards.map(card => {
        const buttons = [...card.querySelectorAll('.answer-header .answer-actions .action-item button')]
            .filter(button => !button.matches('.comment-btn, .report-btn'));
        return {
            author: card.querySelector('.answer-header')?.firstElementChild?.innerText.trim(),
            text: card.querySelector('.answer-content')?.innerText.trim(),
            votes: buttons.map(button => button.innerText.trim())
        };
    })""")
    for record in records:
        votes = record.pop("votes")
        if len(votes) != 2 or any(not re.fullmatch(r"[0-9]+(?:,[0-9]{3})*", count) for count in votes):
            raise ValueError(f"答案赞踩计数无法确认，请核对页面结构：{votes!r}")
        record["likes"], record["dislikes"] = (int(count.replace(",", "")) for count in votes)
    return records


def save_cursor(path, cursor):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        json.dump(cursor, file, ensure_ascii=False)
    Path(file.name).replace(path)


def record_failure(path, page, step, error):
    if path is None:
        return
    record = {"time": datetime.now().astimezone().isoformat(), "step": step,
              "url": page.url.split("?", 1)[0], "error": str(error)}
    if not page.is_closed():
        try:
            record["page"] = page.locator("body").evaluate("""body => ({
                headings: [...body.querySelectorAll('h1,h2,.tab-button.active')]
                    .map(node => node.innerText.trim()).slice(0, 12),
                text: body.innerText.slice(0, 1800),
                answers: body.querySelectorAll('.user-add-answer-item').length
            })""", timeout=2000)
        except Error as snapshot_error:
            record["snapshot_error"] = str(snapshot_error)
    with path.with_name("采集诊断.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_catalog(page, open_page, paper_list, write_paper, stopped, *,
                    check_only=False, sample=False, limit=10, archived=None, progress=None):
    failed_status = []
    cursor = None
    if progress is not None and progress.exists():
        cursor = json.loads(progress.read_text(encoding="utf-8"))
        if cursor.get("finished"):
            print("当前目录遍历已完成。", flush=True)
            return

    def watch(response):
        if response.url.startswith("https://srnz.net/") and response.status in (401, 403, 429, 503):
            failed_status.append(response.status)

    def check_access(tab):
        if failed_status:
            raise stopped(f"收到 HTTP {failed_status[0]}，已停止采集。")
        if not tab.url.startswith(paper_list) or tab.get_by_text(
            re.compile(r"请完成(?:人机|安全)验证|访问过于频繁|访问被拒绝|登录已过期")
        ).first.is_visible():
            raise stopped("目录操作遇到访问限制或页面已离开试卷目录。")

    def click(locator, delay=ACTION_DELAY_MS):
        page.wait_for_timeout(delay)
        check_access(page)
        locator.click(timeout=5000)

    def wait_for_updated_rows(rows, previous_titles):
        try:
            expect(rows).not_to_have_text(previous_titles, timeout=15000)
        except AssertionError as error:
            check_access(page)
            if rows.all_text_contents() != previous_titles:
                raise
            raise TimeoutError("页码已切换，但试卷清单仍未更新。") from error
        rows.first.wait_for(state="visible", timeout=15000)

    context = page.context
    context.on("response", watch)
    try:
        open_page(page, paper_list)
        rows = page.locator(".paper-item .content-text")
        seen = set()
        completed = 0
        for category in (("事业单位",) if sample else ("公务员", "事业单位")):
            if cursor and cursor["category"] != category:
                continue
            category_label = page.locator("label.el-radio").filter(
                has_text=re.compile(f"^\\s*{category}考试\\s*$")
            )
            all_regions = page.locator(".nav-list .nav-all-area")
            if "is-checked" not in (category_label.get_attribute("class") or "").split():
                click(category_label)
            expect(category_label).to_have_class(re.compile(r"\bis-checked\b"))
            expect(all_regions).to_have_text("所有省份" if category == "公务员" else "所有地区")
            regions = page.locator(".nav-list .nav-item")
            category_region = "国考" if category == "公务员" else "联考A类"
            expect(regions.filter(has_text=re.compile(f"^\\s*{category_region}\\s*$"))).to_be_visible()
            region_names = regions.all_text_contents()
            if check_only:
                region_names = region_names[:2]
            elif sample:
                region_names = [random.choice(region_names)]
            checked_pagination = False
            for region in region_names:
                region = region.strip()
                if cursor and cursor["region"] != region:
                    continue
                region_item = regions.filter(has_text=re.compile(f"^\\s*{re.escape(region)}\\s*$"))
                click(region_item)
                expect(region_item).to_have_class(re.compile(r"\bactive\b"))
                page.wait_for_timeout(2000)
                active = page.locator(".el-pagination li.number.active")
                try:
                    rows.first.wait_for(state="visible", timeout=15000)
                except TimeoutError:
                    check_access(page)
                    next_page = page.locator(".el-pagination .btn-next")
                    if (cursor or sample or check_only or rows.count()
                        or active.count() != 1 or active.inner_text(timeout=2000).strip() != "1"
                        or next_page.count() != 1 or not next_page.is_disabled(timeout=2000)):
                        raise
                    print(f"{category}/{region}：目录无条目，且下一页不可用。", flush=True)
                    record_failure(progress, page, f"{category}/{region}", "目录无条目，待复核")
                    continue
                if active.count() and active.inner_text().strip() != "1":
                    first_page = page.locator(".el-pagination li.number").get_by_text("1", exact=True)
                    click(first_page, delay=PAGE_DELAY_MS)
                    expect(active).to_have_text("1")
                    rows.first.wait_for(state="visible", timeout=15000)
                page_number = 1
                if check_only:
                    print(f"{category}/{region}：地区切换成功，首页 {rows.count()} 条", flush=True)
                while True:
                    rows.first.wait_for(state="visible", timeout=15000)
                    titles = rows.all_text_contents()
                    if cursor and page_number == cursor["page"]:
                        if titles != cursor["titles"] or not 0 <= cursor["next_index"] <= len(titles):
                            raise stopped("续采位置的试卷清单已变化，请先核对进度文件。")
                        start_index = cursor["next_index"]
                        cursor = None
                    else:
                        start_index = 0
                    indices = () if check_only or cursor else range(start_index, len(titles))
                    if sample:
                        indices = [random.randrange(len(titles))]
                    for index in indices:
                        title = titles[index]
                        print(f"正在处理 {category}/{region}：{title.strip()}", flush=True)
                        before = set(context.pages)
                        click(rows.nth(index).locator(".."))
                        page.wait_for_timeout(ACTION_DELAY_MS)
                        detail = next((tab for tab in context.pages if tab not in before), page)
                        try:
                            detail.wait_for_url(re.compile(re.escape(paper_list) + r"/detail/\d+"), timeout=15000)
                            check_access(detail)
                            paper_id = int(detail.url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[1])
                            if detail.url not in seen:
                                if not archived or not archived(category, region, paper_id):
                                    collect_paper(detail, write_paper, stopped, category, region)
                                    completed += 1
                                seen.add(detail.url)
                            if progress is not None:
                                save_cursor(progress, dict(category=category, region=region,
                                    page=page_number, next_index=index + 1, titles=titles, paper_id=paper_id))
                        except (Error, AssertionError, stopped) as error:
                            record_failure(progress, detail, f"{category}/{region}/{page_number} {title}", error)
                            if isinstance(error, TimeoutError) and not detail.is_closed():
                                check_access(detail)
                            raise
                        finally:
                            if detail != page:
                                detail.close()
                        if sample or (limit is not None and completed >= limit):
                            return
                        if detail == page:
                            page.wait_for_timeout(ACTION_DELAY_MS)
                            page.go_back(wait_until="domcontentloaded")
                            expect(region_item).to_have_class(re.compile(r"\bactive\b"))
                            rows.first.wait_for(state="visible", timeout=15000)
                            if page_number > 1:
                                expect(active).to_be_visible()
                                while int(active.inner_text()) < page_number:
                                    previous = int(active.inner_text())
                                    previous_titles = rows.all_text_contents()
                                    click(page.locator(".el-pagination .btn-next"), delay=PAGE_DELAY_MS)
                                    expect(active).to_have_text(str(previous + 1))
                                    wait_for_updated_rows(rows, previous_titles)
                            expect(rows).to_have_text([value.strip() for value in titles], timeout=10000)
                            if active.count():
                                expect(active).to_have_text(str(page_number))
                    next_button = page.locator(".el-pagination .btn-next")
                    if not next_button.count() or next_button.is_disabled() or (check_only and checked_pagination):
                        break
                    previous = int(active.inner_text())
                    click(next_button, delay=PAGE_DELAY_MS)
                    expect(active).to_have_text(str(previous + 1))
                    page_number = previous + 1
                    wait_for_updated_rows(rows, titles)
                    if check_only:
                        checked_pagination = True
                        print(f"{category}/{region}：第 {previous} 页 → 第 {previous + 1} 页，清单已更新", flush=True)
                        break
                if cursor:
                    raise stopped("该地区未找到已保存的续采页码。")
                if sample:
                    raise stopped(f"抽中的地区 {region} 没有可见试卷。")
            if check_only and not checked_pagination:
                raise stopped(f"{category}的抽样地区未验证到翻页。")
        if cursor:
            raise stopped("未找到已保存的续采位置。")
        check_access(page)
        if progress is not None:
            save_cursor(progress, {"finished": True})
    except (Error, AssertionError, stopped) as error:
        record_failure(progress, page, "目录遍历", error)
        if isinstance(error, TimeoutError) and not page.is_closed():
            check_access(page)
        raise
    finally:
        context.remove_listener("response", watch)


def collect_paper(page, write_paper, stopped, category="公务员", region="国考"):
    paper_url = page.url.split("?", 1)[0]
    match = re.fullmatch(r"https://srnz\.net/community/studyRoom/paper/detail/(\d+)", paper_url)
    if not match:
        raise stopped("当前页面不是试卷详情。")
    paper_id = int(match[1])
    blocked = []
    def watch(response):
        if response.url.startswith("https://srnz.net/") and response.status in (401, 403, 429, 503):
            blocked.append(response.status)

    page.on("response", watch)

    def check_access():
        if blocked:
            raise stopped(f"收到 HTTP {blocked[0]}，已停止采集。")
        if page.url.split("?", 1)[0] != paper_url or page.get_by_text(
            re.compile(r"请完成(?:人机|安全)验证|访问过于频繁|访问被拒绝|登录已过期")
        ).first.is_visible():
            raise stopped("当前页面已离开试卷或出现访问限制，已停止采集。")

    def read_text(locator):
        content = locator.inner_text(timeout=5000).strip()
        if not content:
            raise stopped("网页中的材料、题干或答案为空。")
        return content

    def select(tab):
        check_access()
        if "active" not in (tab.get_attribute("class") or "").split():
            page.wait_for_timeout(ACTION_DELAY_MS)
            check_access()
            try:
                tab.click(timeout=5000)
            except TimeoutError as error:
                check_access()
                if "intercepts pointer events" not in str(error) or not tab.is_enabled():
                    raise
                tab.press("Enter", timeout=5000)
            expect(tab).to_have_class(re.compile(r"\bactive\b"))
        check_access()

    try:
        title = read_text(page.locator(".paper-header").get_by_role("heading"))
        material_picker = page.locator(".materials-tabs > button")
        question_picker = page.locator(".question-tabs > button")
        try:
            question_picker.first.wait_for(state="visible", timeout=15000)
        except TimeoutError:
            check_access()
            prompt = page.locator(".question-section > .ql-editor").first
            if (question_picker.count() or not prompt.is_visible()
                or prompt.inner_text(timeout=2000).strip()
                or not page.locator(".question-section").get_by_text("暂无参考答案", exact=True).first.is_visible()):
                raise
        material_tabs = material_picker.all()
        question_tabs = question_picker.all()
        if not material_tabs and not question_tabs:
            raise stopped(f"第 {paper_id} 卷没有可识别的材料或题目。")
        if not material_tabs:
            print(f"第 {paper_id} 卷未展示材料页签，继续归档题目与答案。", flush=True)

        materials = []
        material = page.locator(".materials-content")
        for tab in material_tabs:
            switching = "active" not in (tab.get_attribute("class") or "").split()
            previous_material = material.text_content(timeout=5000) if switching else None
            select(tab)
            if switching and previous_material and previous_material.strip():
                try:
                    expect(material).not_to_have_text(previous_material, timeout=5000)
                except AssertionError:
                    check_access()
                    print(f"第 {paper_id} 卷材料 {len(materials) + 1} 与切换前正文相同，待复核。", flush=True)
            try:
                expect(material).not_to_have_text("", timeout=3000)
            except AssertionError:
                check_access()
            content = material.inner_text(timeout=5000).strip()
            images = material.locator("img").evaluate_all(
                "images => images.map(image => image.currentSrc || image.src)"
            )
            image_links = [f"![材料图片](<{url}>)" for url in images
                           if url.startswith(("https://", "http://"))]
            if not content and not image_links:
                check_access()
                content = "采集时此材料页签未显示可读取的内容。"
                print(f"第 {paper_id} 卷材料 {len(materials) + 1} 为空，已标记待复核。", flush=True)
            materials.append("\n\n".join(filter(None, [content, *image_links])))
        if not question_tabs:
            check_access()
            if question_picker.count() or prompt.inner_text(timeout=2000).strip():
                raise stopped(f"第 {paper_id} 卷题目区仍在变化。")
            write_paper(category, region, paper_id, title, materials, [], {})
            return
        questions = []
        references = {}
        question_text = page.locator(".question-section > .question-tabs + .ql-editor")
        container = page.locator(".merged-user-add-answers")
        cards = container.locator(".user-add-answer-item")
        empty_notice = page.locator(".question-section").get_by_text("暂无参考答案", exact=True)

        def snapshot():
            return read_answer_cards(cards)

        for number, tab in enumerate(question_tabs, 1):
            switching = "active" not in (tab.get_attribute("class") or "").split()
            before = snapshot() if switching else None
            previous_card = cards.first.element_handle(timeout=2000) if before else None
            previous = read_text(question_text) if switching else None
            select(tab)
            if switching:
                expect(question_text).not_to_have_text(previous, timeout=7000)
            question = read_text(question_text)
            card_data = snapshot()
            for _ in range(7):
                check_access()
                replaced = previous_card is not None and not previous_card.evaluate("card => card.isConnected")
                if card_data and (card_data != before or replaced):
                    break
                if not card_data and empty_notice.first.is_visible():
                    page.wait_for_timeout(ANSWER_SETTLE_MS)
                    check_access()
                    card_data = snapshot()
                    if not card_data and empty_notice.first.is_visible():
                        break
                page.wait_for_timeout(1000)
                card_data = snapshot()
            else:
                raise stopped(f"第 {number} 题切换后未确认答案更新。")
            initial_count = len(card_data)
            stable = 0
            for _ in range(8 if card_data else 0):
                check_access()
                container.evaluate("node => { node.scrollTop = node.scrollHeight; }")
                page.wait_for_timeout(ANSWER_SETTLE_MS)
                check_access()
                current = snapshot()
                identities = lambda records: [(card["author"], card["text"]) for card in records]
                if identities(current[:len(card_data)]) != identities(card_data):
                    raise stopped(f"第 {number} 题滚动后已读答案发生变化。")
                stable = stable + 1 if len(current) == len(card_data) else 0
                card_data = current
                if stable == 2:
                    break
            else:
                if card_data:
                    raise stopped(f"第 {number} 题滚动后答案仍在变化。")
            check_access()
            for card in card_data:
                if not card.get("author") or not card.get("text"):
                    print(f"第 {number} 题发现字段不完整的答案卡片，已标记。", flush=True)
                    card["author"] = card.get("author") or "页面未提供署名"
                    card["text"] = card.get("text") or "页面未提供可读取的文本正文。"
            print(f"第 {number} 题：{len(card_data)} 份，滚动新增 {len(card_data) - initial_count} 份", flush=True)
            matches = re.findall(r"(?:给定资料|给定材料)[“”\" ]*(\d+)", question)
            if (len(matches) == 1 and 1 <= int(matches[0]) <= len(material_tabs)
                and re.search(r"给定(?:资料|材料)[“”\" ]*\d+[”\"]", question)):
                references[number] = [int(matches[0])]
            questions.append((question, card_data))
        if len({text for text, _ in questions}) != len(questions):
            raise stopped("切换页签后读取到重复题干。")
        if len(set(materials)) != len(materials):
            print(f"第 {paper_id} 卷有相同的材料内容，保留原页签编号并标记待复核。", flush=True)
        write_paper(category, region, paper_id, title, materials, questions, references)
    finally:
        page.remove_listener("response", watch)
