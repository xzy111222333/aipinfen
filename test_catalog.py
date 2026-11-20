import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from playwright.sync_api import Page, sync_playwright

import shenlun
from collect_paper import collect_catalog


CATALOG = r"""
<label class="el-radio is-checked" onclick="category(0)">公务员考试</label>
<label class="el-radio" onclick="category(1)">事业单位考试</label>
<div class="nav-list"></div><main></main><div class="el-pagination"></div>
<script>
let type = 0, number = 1;
function category(value) {
    type = value;
    document.querySelectorAll('label').forEach((label, i) =>
        label.classList.toggle('is-checked', i === type));
    document.querySelector('.nav-list').innerHTML =
        '<div class="nav-all-area">' + (type ? '所有地区' : '所有省份') + '</div>' +
        '<button class="nav-item active" onclick="show(1)">' + (type ? '联考A类' : '国考') + '</button>';
    show(1);
}
function show(value) {
    number = value;
    const id = type * 2 + number;
    document.querySelector('main').innerHTML = '<div class="paper-item" onclick="window.open(location.pathname + \'/detail/' + id + '\')">' +
        '<div class="content-text">试卷 ' + id + '</div></div>';
    document.querySelector('.el-pagination').innerHTML =
        '<ul><li class="number active">' + number + '</li></ul>' +
        '<button class="btn-next" onclick="show(2)" ' + (number === 2 ? 'disabled' : '') + '>下一页</button>';
}
category(0);
</script>
"""


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        playwright = sync_playwright().start()
        cls.addClassCleanup(playwright.stop)
        cls.browser = playwright.chromium.launch(channel="chrome", chromium_sandbox=True)
        cls.addClassCleanup(cls.browser.close)

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.progress = root / "progress.json"
        archive = patch.object(shenlun, "ARCHIVE", root)
        archive.start()
        self.addCleanup(archive.stop)
        self.context = self.browser.new_context(service_workers="block")
        self.addCleanup(self.context.close)
        self.page = self.context.new_page()
        self.details = []
        self.blocked_id = None
        self.empty_id = None
        self.context.route("**/*", self.respond)
        original_wait = Page.wait_for_timeout
        delay = patch.object(Page, "wait_for_timeout", lambda page, _: original_wait(page, 80))
        delay.start()
        self.addCleanup(delay.stop)

    def respond(self, route):
        url = route.request.url
        if url == shenlun.PAPER_LIST:
            route.fulfill(content_type="text/html; charset=utf-8", body=CATALOG)
        elif url.startswith(shenlun.PAPER_LIST + "/detail/"):
            paper_id = int(url.rsplit("/", 1)[1])
            self.details.append(paper_id)
            answers = (
                '<div class="question-section">暂无参考答案</div>'
                if paper_id == self.empty_id else """
                <div class="merged-user-add-answers">
                  <div class="user-add-answer-item">
                    <div class="answer-header"><span>机构甲</span></div>
                    <div class="answer-content">答案正文</div>
                  </div>
                </div>"""
            )
            route.fulfill(status=429 if paper_id == self.blocked_id else 200,
                content_type="text/html; charset=utf-8", body=f"""
                <div class="paper-header"><h1>试卷 {paper_id}</h1></div>
                <div class="materials-tabs"><button class="active">材料1</button></div>
                <div class="materials-content">材料正文</div>
                <div class="question-section">
                  <div class="question-tabs"><button class="active">第1题</button></div>
                  <div class="ql-editor">阅读“给定资料1”。</div>
                </div>
                {answers}""")
        else:
            route.abort()

    def collect(self, limit=None):
        collect_catalog(self.page, shenlun.open_page, shenlun.PAPER_LIST,
                        shenlun.write_paper, shenlun.NavigationStopped, limit=limit,
                        archived=shenlun.already_archived, progress=self.progress)

    def test_resume_page_two_and_finish_both_categories(self):
        self.collect(limit=2)
        self.assertEqual(json.loads(self.progress.read_text(encoding="utf-8"))["page"], 2)
        self.collect()
        self.assertEqual(self.details, [1, 2, 3, 4])
        self.assertEqual(json.loads(self.progress.read_text(encoding="utf-8")), {"finished": True})
        self.assertEqual(len(list(shenlun.ARCHIVE.rglob("第1题.md"))), 4)
        for path in shenlun.ARCHIVE.rglob("第1题.md"):
            text = path.read_text(encoding="utf-8")
            self.assertIn("机构甲", text)
            self.assertIn("(材料1.md)", text)
            self.assertTrue(path.with_name("材料1.md").is_file())

    def test_rate_limit_keeps_checkpoint_and_stops(self):
        self.blocked_id = 2
        with self.assertRaisesRegex(shenlun.NavigationStopped, "HTTP 429"):
            self.collect()
        self.assertEqual(self.details, [1, 2])
        self.assertEqual(json.loads(self.progress.read_text(encoding="utf-8"))["paper_id"], 1)

    def test_explicit_empty_answers_are_archived_and_collection_continues(self):
        self.empty_id = 1
        self.collect(limit=2)
        self.assertEqual(self.details, [1, 2])
        first = shenlun.ARCHIVE / "公务员" / "国考" / "试卷 1" / "第1题.md"
        text = first.read_text(encoding="utf-8")
        self.assertIn("答案（0 份）", text)
        self.assertIn("给定资料1", text)
        self.assertIn("(材料1.md)", text)
