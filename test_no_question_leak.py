import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEED = frozenset("夜市 喷淋 王某 翟某 老皮 长远发展 公众参与科学".split())
ENTITY_PATTERNS = (
    re.compile(r"(?:名叫|笔名[是为]?|化名[为]?|负责人叫)([\u4e00-\u9fff]{2,4})(?=[，。；、\s])"),
    re.compile(r"([A-Z][\u4e00-\u9fff]{1,7}(?:示范园区|实验基地|保障中心|工作站|调查队|产业园))"),
    re.compile(r"[“《]([^”》\r\n]{4,28}(?:工程|整治行动|专题调查|志愿项目|试点平台))[”》]"),
)


def forbidden_entities(source):
    terms = set(SEED)
    for path in source.rglob("*.md"):
        if not re.fullmatch(r"(?:材料|第)\s*\d+(?:题)?\.md", path.name):
            continue
        text = path.read_text(encoding="utf-8-sig")
        if path.name.startswith("第"):
            text = text.partition("## 题目")[2].partition("## 答案")[0]
        for pattern in ENTITY_PATTERNS:
            terms.update(pattern.findall(text))
    return terms


def find_leaks(text, terms):
    literal_text = "\n".join(node.value for node in ast.walk(ast.parse(text))
                             if isinstance(node, ast.Constant) and isinstance(node.value, str))
    return sorted(term for term in terms if term in text or term in literal_text)


class TestQuestionLeak(unittest.TestCase):
    def test_linter_detects_generic_prompt_copying_question_entities(self):
        self.assertEqual(find_leaks('SYSTEM = "夜市和王某"', SEED), ["夜市", "王某"])
        self.assertEqual(find_leaks('SYSTEM = "\\u55b7\\u6dcb"', SEED), ["喷淋"])
        self.assertEqual(find_leaks('SYSTEM = "基于材料核查答卷"', SEED), [])

    def test_v1_contains_no_concrete_question_terms(self):
        directory = ROOT / "scoring" / "v1"
        paths = list(directory.rglob("*.py"))
        if not paths:
            self.skipTest("阶段4尚未编写v1，当前只验证拦截器自身")
        terms = forbidden_entities(ROOT / "试卷")
        for path in sorted(paths):
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertEqual(find_leaks(path.read_text(encoding="utf-8-sig"), terms), [])
