from collections import defaultdict
from itertools import combinations
from statistics import mean


def mean_or_none(values):
    values = list(values)
    return mean(values) if values else None


def variant_result(question, variant, base, scored):
    maximum = question["maximum"]
    diff = base["score"] - scored["score"]
    expectation = variant["expect"]
    relation = expectation["relation"]
    if relation == "equal":
        passed = abs(diff) <= maximum * expectation["max_gap_ratio"] + 1e-8
    elif relation == "lower":
        passed = diff >= maximum * expectation["min_drop_ratio"] - 1e-8
        if expectation.get("strict"):
            passed = passed and diff > 0
    elif relation == "below":
        passed = scored["score"] <= maximum * expectation["max_ratio"] + 1e-8
    elif relation == "overflow":
        passed = scored.get("overflow_excluded") is True and scored["score"] <= base["score"]
    elif relation == "essay_drop":
        passed = scored["band"] > base["band"] or diff >= expectation["min_drop_points"]
    elif relation == "band":
        passed = scored["band"] >= expectation["band_max"]
        if scored["band"] == expectation["band_max"] and "max_in_band" in expectation:
            passed = passed and scored["score"] / maximum <= 0.35 + (0.65 - 0.35) * expectation["max_in_band"]
    else:
        raise ValueError(f"未知对照预期：{relation}")
    if "band_max" in expectation:
        passed = passed and scored["band"] >= expectation["band_max"]
    return {"id": variant["variant_id"], "type": variant["type"], "passed": passed,
            "base_score": base["score"], "score": scored["score"], "diff_ratio": diff / maximum}


def _pass_rate(rows):
    return {"passed": sum(row["passed"] for row in rows), "total": len(rows),
            "rate": mean_or_none(row["passed"] for row in rows)}


def summarize_variants(rows):
    by_type = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)
    return {
        "M2": _pass_rate([row for row in rows if row["type"].startswith("V")]),
        "M3": _pass_rate([row for row in rows if not row["type"].startswith("V")]),
        "by_type": {key: _pass_rate(group) for key, group in sorted(by_type.items())},
    }


def summarize_stability(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["question_id"], r["answer_id"]].append(r)
    fixed, rebuilt = [], []
    for (question_id, answer_id), grades in sorted(grouped.items()):
        slot = grades[0]
        ordinary = [r["score"] for r in grades if r["rubric_round"] == 0]
        new_rubrics = [r["score"] for r in grades if r["rubric_round"] > 0]
        if len(ordinary) != 5 or len(new_rubrics) not in {0, 2}:
            raise ValueError(f"稳定性样本不完整：{answer_id}")
        span = max(ordinary) - min(ordinary)
        fixed.append({"question_id": question_id, "answer_id": answer_id, "spread": span,
                      "ratio": span / slot["maximum"], "passed": span <= (2 if slot["essay"] else slot["maximum"] * 0.05)})
        if new_rubrics:
            compared = [ordinary[0], *new_rubrics]
            difference = max(compared) - min(compared)
            rebuilt.append({"question_id": question_id, "answer_id": answer_id, "spread": difference,
                            "ratio": difference / slot["maximum"], "passed": difference <= slot["maximum"] * 0.10})
    return {"fixed": _pass_rate(fixed), "rebuilt": _pass_rate(rebuilt), "fixed_details": fixed, "rebuilt_details": rebuilt}


def compare_real(rows, human=()):
    scores = [r for r in rows if r["prediction"] is not None]
    pair_records = []
    for left, right in combinations(scores, 2):
        if left["paper"] != right["paper"] or left["recorded"] == right["recorded"]:
            continue
        pair_records.append({"a": left["path"], "b": right["path"], "source": "real",
                             "passed": (left["recorded"] - right["recorded"]) * (left["prediction"] - right["prediction"]) > 0})
    pair_records.extend(human)
    differences = [r["prediction"] - r["recorded"] for r in scores]
    return {"M4": {**_pass_rate(pair_records), "pairs": pair_records},
            "M5": {"n": len(scores), "mae": mean_or_none(abs(d) for d in differences), "bias": mean_or_none(differences)}}
