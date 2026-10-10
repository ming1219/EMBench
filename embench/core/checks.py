"""
检查项库（声明式 verifier 的积木）
================================
task.toml 里每个 [[verifier.checks]] 对应这里一个注册过的检查函数：

    [[verifier.checks]]
    id = "trajectory"
    type = "tool_calls_match"
    ordered = true
    calls = [{ tool = "calculateKJ", args = { smiles = "..." } }]

所有检查共享的通用字段：
    id          唯一标识（缺省 = type）
    type        检查类型（见 CHECKS 注册表）
    required    是否计入 pass 判定（缺省按类型：call_budget=false，其余 true）
    threshold   各维度分数 ≥ threshold 才算该检查通过（缺省 1.0）
    gate        哪些维度参与 pass 判定（缺省 = 该检查产出的全部维度）
    description 人读说明

每个检查返回若干「维度分数」∈[0,1]；None 表示数据不足而跳过（如旧轨迹没录工具结果）。
维度约定：
    l1_tool     工具选择/顺序        l2_args   参数正确/合规
    l3_value    工具输出数值正确      coverage  候选覆盖完整度
    consistency 下游输入与上游输出自洽 answer    最终回答质量
    efficiency  调用效率（软指标）

新增检查：写一个函数并用 @register(...) 注册，声明参数表；checker 会据此校验 task.toml
里的拼写错误与缺参（未知参数直接报错，避免静默失效）。
"""
from __future__ import annotations

import importlib.util
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .normalize import (
    DEFAULT_ARG_RULES,
    HAS_RDKIT,
    canonical_smiles,
    coerce_list,
    coerce_number,
    extract_smiles,
    get_arg,
    is_valid_smiles,
    numbers_close,
    normalize_value,
    parse_json,
    values_equal,
)
from .trajectory import Step, Trajectory

COMMON_PARAMS = {"id", "type", "required", "threshold", "gate", "description"}


@dataclass
class CheckSpec:
    fn: Callable
    params: dict  # name -> required(bool)
    dimensions: tuple
    required_default: bool = True


@dataclass
class CheckResult:
    id: str
    type: str
    scores: dict
    passed: bool | None
    required: bool
    detail: str = ""
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "required": self.required,
            "passed": self.passed,
            "scores": self.scores,
            "detail": self.detail,
            **({"data": self.data} if self.data else {}),
        }


CHECKS: dict[str, CheckSpec] = {}


def register(name: str, *, params: dict, dimensions: tuple, required_default: bool = True):
    def deco(fn):
        CHECKS[name] = CheckSpec(fn, params, dimensions, required_default)
        return fn
    return deco


# ================================================================
#  工具函数
# ================================================================

def _arg_rule(key: str, overrides: dict | None) -> str | None:
    if overrides and key in overrides:
        return overrides[key]
    return DEFAULT_ARG_RULES.get(key.lower())


def args_match(actual: dict, expected: dict, arg_rules: dict | None = None) -> tuple[bool, str]:
    """expected 中声明的每个参数都必须匹配；actual 多出的参数默认不管（见 forbid_args）。"""
    for key, exp_val in (expected or {}).items():
        present, act_val = get_arg(actual, key)
        if not present:
            return False, f"缺参数 {key}"
        if not values_equal(act_val, exp_val, _arg_rule(key, arg_rules)):
            return False, f"{key}: {act_val!r} ≠ {exp_val!r}"
    return True, ""


def _call_matches(step: Step, exp: dict) -> tuple[bool, str]:
    if step.tool != exp["tool"]:
        return False, f"工具 {step.tool} ≠ {exp['tool']}"
    ok, why = args_match(step.args, exp.get("args", {}), exp.get("arg_rules"))
    if not ok:
        return False, why
    for key in exp.get("forbid_args", []):
        if get_arg(step.args, key)[0]:
            return False, f"不应传参数 {key}"
    return True, ""


def _subsequence(steps: list[Step], expected: list[dict], pred) -> list[int | None]:
    """有序子序列匹配：贪心最早匹配即最优。返回每个期望项匹配到的 step 下标（None=未匹配）。"""
    out, pos = [], 0
    for exp in expected:
        hit = None
        for i in range(pos, len(steps)):
            if pred(steps[i], exp):
                hit = i
                break
        out.append(hit)
        if hit is None:
            break  # 有序：前一项没匹配上，后面全部视为未匹配
        pos = hit + 1
    return out + [None] * (len(expected) - len(out))


def _unordered(steps: list[Step], expected: list[dict], pred) -> list[int | None]:
    used, out = set(), []
    for exp in expected:
        hit = next((i for i, s in enumerate(steps) if i not in used and pred(s, exp)), None)
        if hit is not None:
            used.add(hit)
        out.append(hit)
    return out


def _result_obj(step: Step) -> Any:
    return parse_json(step.result)


def _pick_record(obj: Any, match_smiles: str | None) -> dict:
    """工具结果可能是 dict 或 list[dict]（批量）；按 smiles 选记录，否则取第一条。"""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        recs = [r for r in obj if isinstance(r, dict)]
        if match_smiles:
            target = canonical_smiles(match_smiles)
            for r in recs:
                if canonical_smiles(r.get("smiles", "")) == target:
                    return r
        return recs[0] if recs else {}
    return {}


def _ratio(n: int, d: int) -> float:
    return n / d if d else 0.0


# ================================================================
#  L1 / L2：轨迹匹配
# ================================================================

@register(
    "tool_calls_match",
    params={"calls": True, "ordered": False, "allow_extra": False},
    dimensions=("l1_tool", "l2_args"),
)
def tool_calls_match(traj: Trajectory, p: dict, task) -> CheckResult:
    """期望调用序列是否出现在轨迹里（工具名 → l1；工具名+参数 → l2）。

    ordered=true（缺省）按子序列匹配；allow_extra=true（缺省）允许冗余调用，
    冗余数记在 data.redundancy；allow_extra=false 时要求调用数恰好相等。
    """
    expected = p["calls"]
    ordered = p.get("ordered", True)
    allow_extra = p.get("allow_extra", True)
    matcher = _subsequence if ordered else _unordered

    name_hits = matcher(traj.steps, expected, lambda s, e: s.tool == e["tool"])
    full_hits = matcher(traj.steps, expected, lambda s, e: _call_matches(s, e)[0])

    l1 = _ratio(sum(h is not None for h in name_hits), len(expected))
    l2 = _ratio(sum(h is not None for h in full_hits), len(expected))
    exact = len(traj.steps) == len(expected) and l2 == 1.0

    # 第一个未匹配项的原因，便于排错
    reasons = []
    for exp, hit in zip(expected, full_hits):
        if hit is None:
            cands = [s for s in traj.steps if s.tool == exp["tool"]]
            why = _call_matches(cands[0], exp)[1] if cands else "未调用"
            reasons.append(f"{exp['tool']}: {why}")
            break
    matched = sum(h is not None for h in full_hits)
    if not allow_extra and not exact:
        reasons.append(f"要求恰好 {len(expected)} 次调用，实际 {len(traj.steps)} 次")
    return CheckResult(
        id="", type="", scores={"l1_tool": l1, "l2_args": l2},
        passed=False if (not allow_extra and not exact) else None, required=True,
        detail="; ".join(reasons) or ("exact" if exact else f"子序列匹配，冗余 {len(traj.steps) - matched}"),
        data={"exact": exact, "redundancy": len(traj.steps) - matched},
    )


@register(
    "tools_used",
    params={"all_of": False, "none_of": False},
    dimensions=("l1_tool",),
)
def tools_used(traj: Trajectory, p: dict, task) -> CheckResult:
    """不关心顺序与参数，只看某些工具是否被调用过 / 是否没被调用。"""
    used = set(traj.tool_names)
    all_of, none_of = p.get("all_of", []), p.get("none_of", [])
    hits = [t in used for t in all_of] + [t not in used for t in none_of]
    missing = [t for t in all_of if t not in used] + [f"!{t}" for t in none_of if t in used]
    return CheckResult("", "", {"l1_tool": _ratio(sum(hits), len(hits)) if hits else 1.0}, None, True,
                       detail=f"缺 {missing}" if missing else "ok")




def _eval_constraint(args: dict, key: str, spec: dict) -> tuple[bool, str]:
    present, val = get_arg(args, key)
    if spec.get("absent"):
        return (not present or val is None), f"{key} 应缺省"
    if spec.get("present") and not present:
        return False, f"缺参数 {key}"
    if not present:
        if "default" not in spec:
            return False, f"缺参数 {key}"
        val = spec["default"]
    rule = spec.get("rule") or DEFAULT_ARG_RULES.get(key.lower())
    for op, target in spec.items():
        if op in ("present", "absent", "default", "rule"):
            continue
        if op == "eq":
            ok = values_equal(val, target, rule)
        elif op == "ne":
            ok = not values_equal(val, target, rule)
        elif op in ("le", "ge", "lt", "gt"):
            v = coerce_number(val)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                return False, f"{key}={val!r} 非数值"
            ok = {"le": v <= target, "ge": v >= target, "lt": v < target, "gt": v > target}[op]
        elif op == "in":
            ok = any(values_equal(val, t, rule) for t in target)
        elif op in ("set_eq", "subset_of"):
            r = rule if rule in ("smiles_set", "str_set") else "str_set"
            a, t = normalize_value(val, r), normalize_value(target, r)
            ok = (a == t) if op == "set_eq" else (a <= t)
        else:
            raise ValueError(f"未知约束算子 {op}")
        if not ok:
            return False, f"{key}={val!r} 不满足 {op} {target!r}"
    return True, ""


@register(
    "arg_constraints",
    params={"tool": True, "constraints": True, "which": False},
    dimensions=("l2_args",),
)
def arg_constraints(traj: Trajectory, p: dict, task) -> CheckResult:
    """对某工具的参数做约束判定（用于随机输出、无法给定精确轨迹的生成类任务）。

    constraints = { num = { eq = 3 }, use_em = { eq = true, default = true },
                    max_routes = { le = 3 }, density = { absent = true } }
    which = "first"（缺省）| "all"（每次调用都要满足）| "any"（任一次满足即可）
          | "sum"（数值参数跨调用求和后判定，如分两类各生成 2 个 → num 合计 4）
    """
    calls = traj.calls(p["tool"])
    if not calls:
        return CheckResult("", "", {"l2_args": 0.0}, None, True, detail=f"{p['tool']} 未调用")
    which = p.get("which", "first")
    if which == "sum":
        merged = {}
        for step in calls:
            for k, v in step.args.items():
                n = coerce_number(v)
                if isinstance(n, (int, float)) and not isinstance(n, bool) and isinstance(merged.get(k, 0), (int, float)):
                    merged[k] = merged.get(k, 0) + n
                else:
                    merged.setdefault(k, v)
        targets = [Step(p["tool"], merged)]
    else:
        targets = calls[:1] if which == "first" else calls
    per_call, fails = [], []
    for step in targets:
        res = [_eval_constraint(step.args, k, spec) for k, spec in p["constraints"].items()]
        per_call.append(_ratio(sum(ok for ok, _ in res), len(res)))
        fails += [why for ok, why in res if not ok]
    score = max(per_call) if which == "any" else sum(per_call) / len(per_call)
    return CheckResult("", "", {"l2_args": score}, None, True, detail="; ".join(dict.fromkeys(fails)) or "ok")


# ================================================================
#  L3：工具输出
# ================================================================

@register(
    "tool_result_values",
    params={"tool": True, "expected": True, "rel_tol": False, "abs_tol": False,
            "match_smiles": False, "which": False},
    dimensions=("l3_value",),
)
def tool_result_values(traj: Trajectory, p: dict, task) -> CheckResult:
    """工具返回值的字段是否与参考值一致（数值走相对容差，字符串精确比较）。

    which = "any"（缺省，任一次调用的结果达标即可）| "last" | "first"
    """
    calls = traj.calls(p["tool"])
    if not calls:
        return CheckResult("", "", {"l3_value": 0.0}, None, True, detail=f"{p['tool']} 未调用")
    with_result = [s for s in calls if s.has_result]
    if not with_result:
        return CheckResult("", "", {"l3_value": None}, None, True, detail="轨迹未记录工具结果，跳过")
    which = p.get("which", "any")
    if which == "first":
        with_result = with_result[:1]
    elif which == "last":
        with_result = with_result[-1:]
    rel, abs_ = p.get("rel_tol", 1e-2), p.get("abs_tol", 1e-6)
    expected = p["expected"]

    best, best_fail, best_nulls = -1.0, [], 0
    for step in with_result:
        rec = _pick_record(_result_obj(step), p.get("match_smiles"))
        ok_n, fails, nulls = 0, [], 0
        for f, ev in expected.items():
            av = rec.get(f)
            if isinstance(ev, (int, float)) and not isinstance(ev, bool):
                ok = numbers_close(av, ev, rel_tol=rel, abs_tol=abs_)
                nulls += not _is_real_number(av)
            else:
                ok = av == ev
                nulls += av is None
            ok_n += ok
            if not ok:
                fails.append(f"{f}: {av!r} vs {ev!r}")
        score = _ratio(ok_n, len(expected))
        if score > best:
            best, best_fail, best_nulls = score, fails, nulls
    invalid = best < 1 and best_nulls > 0
    return CheckResult("", "", {"l3_value": best}, None, True,
                       detail=("证据无效（关键字段为 null 或非数值）；" if invalid else "") + ("; ".join(best_fail[:4]) or "ok"),
                       data={"invalid_evidence": True} if invalid else {})


@register(
    "generation_output",
    params={"tool": False, "min_count": True, "required_properties": False},
    dimensions=("l3_value",),
)
def generation_output(traj: Trajectory, p: dict, task) -> CheckResult:
    """生成类工具的输出：真实生成的分子数 ≥ min_count，且至少 min_count 个生成的分子有全部所需性质的数值。

    只认真实证据：null、空串、解析不了的 SMILES 不算生成了分子；性质值必须是有限数值，
    并且记录里的 smiles 必须是本次生成的分子。用 null 凑数量或凑字段 → 判失败并标记「证据无效」。
    """
    tool = p.get("tool", "generateAndPredict")
    calls = traj.calls(tool)
    if not calls:
        return CheckResult("", "", {"l3_value": 0.0}, None, True, detail=f"{tool} 未调用")
    with_result = [s for s in calls if s.has_result]
    if not with_result:
        return CheckResult("", "", {"l3_value": None}, None, True, detail="轨迹未记录工具结果，跳过")
    props = list(p.get("required_properties") or [])
    need = p["min_count"]
    best, best_detail, best_invalid = -1.0, "", False
    for step in with_result:
        obj = _result_obj(step)
        if not isinstance(obj, dict):
            continue
        valid, bad = _valid_molecules(obj.get("generated_molecules"))
        n_valid = len(set(valid))
        count_ok = n_valid >= need
        null_props, n_with_props = False, 0
        if props:
            results = obj.get("results") if isinstance(obj.get("results"), list) else []
            covered = set()
            for r in results:
                if not isinstance(r, dict):
                    continue
                if any(not _is_real_number(r.get(k)) for k in props if k in r):
                    null_props = True
                smi = r.get("smiles")
                if (isinstance(smi, str) and canonical_smiles(smi) in set(valid)
                        and all(_is_real_number(r.get(k)) for k in props)):
                    covered.add(canonical_smiles(smi))
            n_with_props = len(covered)
            props_ok = n_with_props >= need
        else:
            props_ok = True
        score = (count_ok + props_ok) / 2
        invalid = (not count_ok and bad > 0) or (not props_ok and null_props)
        detail = f"有效生成 {n_valid} 个(需≥{need})"
        if bad:
            detail += f"，另有 {bad} 条无效（null/空/无法解析）"
        if props:
            detail += f"；有全部所需性质数值的分子 {n_with_props} 个(需≥{need})"
        if score > best:
            best, best_detail, best_invalid = score, detail, invalid
    if best < 0:
        return CheckResult("", "", {"l3_value": 0.0}, None, True, detail="工具结果不是有效的 JSON 对象")
    if best_invalid:
        best_detail = "证据无效：" + best_detail
    return CheckResult("", "", {"l3_value": best}, None, True, detail=best_detail,
                       data={"invalid_evidence": True} if best_invalid and best < 1 else {})


def _is_real_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _valid_molecules(items) -> tuple[list, int]:
    """(有效分子的 canonical SMILES 列表, 无效条目数)。null、空串、非字符串、RDKit 解析不了的都算无效。"""
    valid, bad = [], 0
    for s in items if isinstance(items, list) else []:
        if isinstance(s, str) and s.strip() and (is_valid_smiles(s.strip()) if HAS_RDKIT else True):
            valid.append(canonical_smiles(s))
        else:
            bad += 1
    return valid, bad


def _generated(traj: Trajectory, tool: str, field_name: str) -> tuple:
    """(本次真实生成的分子集合 | None, 无效条目数)。调了但没录结果 → (None, 0)，表示无法判定。"""
    calls = [s for s in traj.calls(tool) if s.has_result]
    if traj.calls(tool) and not calls:
        return None, 0
    out, bad = set(), 0
    for step in calls:
        obj = _result_obj(step)
        if isinstance(obj, dict):
            valid, n_bad = _valid_molecules(obj.get(field_name) or [])
            out.update(valid)
            bad += n_bad
    return out, bad


def _generated_set(traj: Trajectory, tool: str, field_name: str) -> set | None:
    return _generated(traj, tool, field_name)[0]


@register(
    "candidate_coverage",
    params={"source_tool": False, "source_field": False, "downstream": True, "min_candidates": False},
    dimensions=("coverage", "consistency"),
)
def candidate_coverage(traj: Trajectory, p: dict, task) -> CheckResult:
    """生成→下游流水线：每个生成的候选是否都被下游工具处理（coverage），
    下游工具的输入是否都来自生成结果而非编造（consistency）。

    downstream = { calculateKJ = "smiles", predictRetrosynthesis = "smiles" }
    coverage    = |G ∩ 每个下游工具覆盖集合的交集| / |G|
    consistency = 下游调用中输入 ∈ G 的比例
    """
    G, bad = _generated(traj, p.get("source_tool", "generateAndPredict"), p.get("source_field", "generated_molecules"))
    if G is None:
        return CheckResult("", "", {"coverage": None, "consistency": None}, None, True, detail="未记录生成结果，跳过")
    min_c = p.get("min_candidates", 1)
    if len(G) < min_c and bad:
        return CheckResult("", "", {"coverage": 0.0, "consistency": 0.0}, None, True,
                           detail=f"证据无效：生成结果中有效分子 {len(G)} 个，另有 {bad} 条 null/无法解析",
                           data={"invalid_evidence": True})
    complete = set(G)
    total, inside = 0, 0
    per_tool = {}
    for tool, key in p["downstream"].items():
        covered = set()
        for step in traj.calls(tool):
            for s in coerce_list(get_arg(step.args, key)[1]):
                c = canonical_smiles(str(s))
                total += 1
                inside += c in G
                covered.add(c)
        per_tool[tool] = len(G & covered)
        complete &= covered
    coverage = _ratio(len(complete), len(G)) if len(G) >= min_c else 0.0
    consistency = _ratio(inside, total)
    detail = f"|G|={len(G)} 完整={len(complete)} " + " ".join(f"{t}={n}" for t, n in per_tool.items())
    return CheckResult("", "", {"coverage": coverage, "consistency": consistency}, None, True, detail=detail,
                       data={"generated": len(G), "per_tool": per_tool})


# ================================================================
#  最终回答
# ================================================================

@register(
    "answer_contains",
    params={"any_of": False, "all_of": False, "none_of": False, "min_length": False},
    dimensions=("answer",),
)
def answer_contains(traj: Trajectory, p: dict, task) -> CheckResult:
    """关键词检查。注意子串匹配的局限：「不推荐」也包含「推荐」——
    用 none_of 排除拒答/否定说法；要判断「推荐了什么」请用 answer_mentions_candidate。"""
    ans = traj.final_answer if isinstance(traj.final_answer, str) else str(traj.final_answer or "")
    if traj.error or len(ans) < p.get("min_length", 0):
        return CheckResult("", "", {"answer": 0.0}, None, True, detail="无有效回答")
    hit_forbidden = [k for k in p.get("none_of", []) if k in ans]
    if hit_forbidden:
        return CheckResult("", "", {"answer": 0.0}, None, True, detail=f"回答含否定/拒答说法 {hit_forbidden}")
    any_of, all_of = p.get("any_of", []), p.get("all_of", [])
    conds = ([any(k in ans for k in any_of)] if any_of else []) + [k in ans for k in all_of]
    score = _ratio(sum(conds), len(conds)) if conds else 1.0
    return CheckResult("", "", {"answer": score}, None, True, detail="ok" if score == 1 else "关键词缺失")


RECOMMEND_KEYWORDS = ("推荐", "建议", "首选", "最优", "最佳", "优先")
# 含这些说法的小句视为否定/暂缓：其中的分子是「被排除的」，不是「被推荐的」
NEGATION_MARKERS = ("不推荐", "不建议", "不予", "无法推荐", "不能推荐", "不宜", "暂缓", "暂不",
                    "证据不足", "无法作出", "无法给出", "排除", "淘汰", "避免选", "不考虑")
_CLAUSE_SPLIT = re.compile(r"[。！？!?；;\n，,]")


def recommended_smiles(answer, keywords=RECOMMEND_KEYWORDS, negations=NEGATION_MARKERS) -> set:
    """从最终回答里找出「被推荐的分子」（canonical SMILES 集合）。

    把回答切成小句；含推荐词且不含否定/暂缓说法的小句是推荐句，
    推荐句里出现的分子即被推荐的分子。只是被列举、被比较或被排除的分子不算。
    """
    if not isinstance(answer, str):
        return set()
    out = set()
    for clause in _CLAUSE_SPLIT.split(answer):
        if any(k in clause for k in keywords) and not any(n in clause for n in negations):
            out |= extract_smiles(clause)
    return out


@register(
    "answer_recommends_candidate",
    params={"source_tool": False, "source_field": False, "min_count": False,
            "keywords": False, "negations": False},
    dimensions=("answer",),
)
def answer_recommends_candidate(traj: Trajectory, p: dict, task) -> CheckResult:
    """最终回答必须明确推荐至少 min_count 个分子，且被推荐的分子**全部**来自本次生成结果。

    比 answer_mentions_candidate 严格：后者只看回答里有没有提到生成过的分子，
    所以「候选包括 A，建议暂缓」或「候选包括 A，推荐生成集外的 B」都能蒙混过关；
    本检查只看推荐句里的分子：暂缓/拒答 → 没有被推荐的分子 → 失败；推荐了集外分子 → 失败。
    """
    G = _generated_set(traj, p.get("source_tool", "generateAndPredict"), p.get("source_field", "generated_molecules"))
    if G is None:
        return CheckResult("", "", {"answer": None}, None, True, detail="未记录生成结果，无法核对推荐对象")
    if traj.error:
        return CheckResult("", "", {"answer": 0.0}, None, True, detail="运行出错，无有效回答")
    rec = recommended_smiles(traj.final_answer, tuple(p.get("keywords", RECOMMEND_KEYWORDS)),
                             tuple(p.get("negations", NEGATION_MARKERS)))
    outside = rec - G
    need = p.get("min_count", 1)
    if outside:
        detail = f"推荐了 {len(outside)} 个不在本次生成结果中的分子: {sorted(outside)[:2]}"
        return CheckResult("", "", {"answer": 0.0}, None, True, detail=detail, data={"recommended": sorted(rec)})
    if len(rec) < need:
        detail = "回答没有明确推荐任何分子（暂缓、拒答或只是列举）" if not rec else f"只推荐了 {len(rec)} 个，需要 ≥{need}"
        return CheckResult("", "", {"answer": 0.0}, None, True, detail=detail, data={"recommended": sorted(rec)})
    return CheckResult("", "", {"answer": 1.0}, None, True, detail=f"推荐了 {len(rec)} 个生成过的候选",
                       data={"recommended": sorted(rec)})


@register(
    "answer_mentions_candidate",
    params={"source_tool": False, "source_field": False, "min_count": False},
    dimensions=("answer",),
)
def answer_mentions_candidate(traj: Trajectory, p: dict, task) -> CheckResult:
    """最终回答必须点名（给出 SMILES）至少 min_count 个本次真实生成过的候选。

    比关键词可靠：拒答、空泛推荐、推荐编造的分子都过不了。
    没记录生成结果时无法判定（缺证据 → 默认判不通过，见 verifier）。
    """
    G = _generated_set(traj, p.get("source_tool", "generateAndPredict"), p.get("source_field", "generated_molecules"))
    if G is None:
        return CheckResult("", "", {"answer": None}, None, True, detail="未记录生成结果，无法核对回答中的候选")
    if traj.error:
        return CheckResult("", "", {"answer": 0.0}, None, True, detail="运行出错，无有效回答")
    found = extract_smiles(traj.final_answer)
    named = found & G
    need = p.get("min_count", 1)
    fabricated = found - G
    detail = f"回答点名了 {len(named)} 个生成过的候选（需要 ≥{need}）"
    if fabricated:
        detail += f"，另有 {len(fabricated)} 个不在生成结果中的分子"
    return CheckResult("", "", {"answer": 1.0 if len(named) >= need else 0.0}, None, True, detail=detail)


@register(
    "answer_smiles",
    params={"expected": True, "mode": False},
    dimensions=("answer",),
)
def answer_smiles(traj: Trajectory, p: dict, task) -> CheckResult:
    """最终回答里是否给出了正确分子（canonical SMILES 比较，写法不同不扣分）。"""
    found = extract_smiles(traj.final_answer)
    exp = [canonical_smiles(s) for s in coerce_list(p["expected"])]
    hits = [e in found for e in exp]
    score = (1.0 if any(hits) else 0.0) if p.get("mode") == "any" else _ratio(sum(hits), len(hits))
    return CheckResult("", "", {"answer": score}, None, True, detail="ok" if score == 1 else f"回答中未找到 {len(hits) - sum(hits)} 个期望分子")


# ================================================================
#  效率（软指标）
# ================================================================

@register("call_budget", params={"max_calls": True}, dimensions=("efficiency",), required_default=False)
def call_budget(traj: Trajectory, p: dict, task) -> CheckResult:
    n = len(traj.steps)
    score = 1.0 if n <= p["max_calls"] else p["max_calls"] / n
    return CheckResult("", "", {"efficiency": score}, None, False, detail=f"{n} 次调用 / 预算 {p['max_calls']}")


# ================================================================
#  逃生口：任务自带 Python 检查
# ================================================================

@register("python", params={"path": True, "function": False, "dimensions": False}, dimensions=())
def python_check(traj: Trajectory, p: dict, task) -> CheckResult:
    """调用 <task_dir>/<path> 中的 verify(traj, task, params) → {dimension: score} 或 CheckResult 字段 dict。"""
    path = task.dir / p["path"]
    spec = importlib.util.spec_from_file_location(f"embench_task_check_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = getattr(mod, p.get("function", "verify"))(traj, task, p)
    if "scores" in out:
        return CheckResult("", "", out["scores"], None, True, detail=out.get("detail", ""), data=out.get("data", {}))
    return CheckResult("", "", dict(out), None, True)


# ================================================================
#  执行单个检查
# ================================================================

def run_check(traj: Trajectory, spec: dict, task) -> CheckResult:
    ctype = spec["type"]
    cs = CHECKS[ctype]
    params = {k: v for k, v in spec.items() if k not in COMMON_PARAMS}
    try:
        res = cs.fn(traj, params, task)
    except Exception as e:  # 检查本身崩溃 = 判失败并记录，绝不让单题拖垮整个 job
        dims = cs.dimensions or tuple(spec.get("dimensions", ("custom",)))
        res = CheckResult("", ctype, {d: 0.0 for d in dims}, None, True, detail=f"检查异常: {type(e).__name__}: {e}")
    forced_fail = res.passed is False  # 检查函数可直接判失败（如 allow_extra=false 时调用数不符）
    res.id = spec.get("id", ctype)
    res.type = ctype
    res.required = spec.get("required", cs.required_default)
    threshold = spec.get("threshold", 1.0)
    gate = spec.get("gate") or list(res.scores)
    gated = [res.scores[d] for d in gate if res.scores.get(d) is not None]
    res.passed = None if not gated else (all(s >= threshold - 1e-9 for s in gated) and not forced_fail)
    return res
