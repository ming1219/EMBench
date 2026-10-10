"""
参数 / 结果归一化
================
评测侧要跟随工具本身的容错：同一分子不同 SMILES 写法、LLM 把数组参数传成
JSON 字符串、分子名别名等，都不应判错。所有归一化集中在这里，便于单测与复用。
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

try:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except Exception:  # rdkit 不可用时退化为字符串比较
    Chem = None
    HAS_RDKIT = False


# ----------------------------------------------------------------
#  基础类型
# ----------------------------------------------------------------

def canonical_smiles(s: Any) -> Any:
    """RDKit canonical；解析失败或无 rdkit 时返回去空白原串。"""
    if not isinstance(s, str):
        return s
    s = s.strip()
    if not s or not HAS_RDKIT:
        return s
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return s
    return Chem.MolToSmiles(mol) or s


def is_valid_smiles(s: str) -> bool:
    if not HAS_RDKIT or not isinstance(s, str) or not s:
        return False
    return Chem.MolFromSmiles(s) is not None


def coerce_list(val: Any) -> list:
    """把 list / JSON 字符串 '["a","b"]' / 逗号分隔串 / 单值 统一成 list（镜像工具的 _parse_list）。"""
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        return list(val)
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            return list(parsed) if isinstance(parsed, list) else [parsed]
        except (json.JSONDecodeError, ValueError):
            return [p.strip() for p in s.split(",") if p.strip()]
    return [val]


def coerce_bool(val: Any) -> Any:
    if isinstance(val, str) and val.strip().lower() in ("true", "false"):
        return val.strip().lower() == "true"
    return val


def coerce_number(val: Any) -> Any:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        try:
            return float(val.strip())
        except ValueError:
            return val
    return val


def alias_name(name: Any) -> Any:
    """nameToSmiles 的 name 归一：CL20 / CL-20 / cl_20 视为同一名。"""
    if not isinstance(name, str):
        return name
    return re.sub(r"[-_\s]", "", name).lower()


def parse_json(raw: Any) -> Any:
    """工具结果多为 JSON 字符串；能解析就解析，否则原样返回。"""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw
    return raw


def numbers_close(a: Any, b: Any, rel_tol: float = 1e-3, abs_tol: float = 1e-6) -> bool:
    a, b = coerce_number(a), coerce_number(b)
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)


# ----------------------------------------------------------------
#  按参数名推断归一规则（可在 task.toml 用 arg_rules 覆盖）
# ----------------------------------------------------------------

_NUMERIC = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")

DEFAULT_ARG_RULES = {
    "smiles": "smiles",
    "smiles_list": "smiles_set",
    "name": "alias",
    "properties": "str_set",
    "formula": "str",
}


def normalize_value(val: Any, rule: str | None) -> Any:
    """按规则归一单个值，返回可直接 == 比较的形式（数值除外，见 values_equal）。"""
    if rule == "smiles":
        return canonical_smiles(val)
    if rule == "smiles_set":
        return frozenset(canonical_smiles(str(x)) for x in coerce_list(val))
    if rule == "alias":
        return alias_name(val)
    if rule == "str_set":
        return frozenset(str(x).strip().lower() for x in coerce_list(val))
    if rule == "str":
        return val.strip() if isinstance(val, str) else val
    if rule == "number":
        return coerce_number(val)
    # 默认：字符串小写，"true"/"false" → bool，数字串 → 数值
    val = coerce_bool(val)
    if isinstance(val, str):
        if _NUMERIC.match(val.strip()):
            return float(val)
        return val.strip().lower()
    if isinstance(val, list):
        return [normalize_value(v, None) for v in val]
    return val


def values_equal(actual: Any, expected: Any, rule: str | None = None, rel_tol: float = 1e-3) -> bool:
    a = normalize_value(actual, rule)
    e = normalize_value(expected, rule)
    if isinstance(e, (int, float)) and not isinstance(e, bool):
        return numbers_close(a, e, rel_tol=rel_tol)
    if isinstance(e, list) and isinstance(a, list):
        # 列表默认顺序无关
        if len(a) != len(e):
            return False
        remaining = list(a)
        for item in e:
            hit = next((i for i, x in enumerate(remaining) if values_equal(x, item, None, rel_tol)), None)
            if hit is None:
                return False
            remaining.pop(hit)
        return True
    return a == e


def get_arg(args: dict, key: str, default: Any = None) -> tuple[bool, Any]:
    """大小写无关取参数，返回 (是否存在, 值)。"""
    if not isinstance(args, dict):
        return False, default
    if key in args:
        return True, args[key]
    kl = key.lower()
    for k, v in args.items():
        if isinstance(k, str) and k.lower() == kl:
            return True, v
    return False, default


# ----------------------------------------------------------------
#  从自由文本中抽 SMILES（判最终回答用）
# ----------------------------------------------------------------

_SMILES_TOKEN = re.compile(r"[A-Za-z0-9@+\-\[\]()=#$/\\%.]{2,}")


def extract_smiles(text: Any) -> set:
    """抽取文本里所有能被 RDKit 解析的 token，返回 canonical 集合。

    纯字母短词（如 "TNT"、"SMILES"）也可能被 RDKit 当成合法 SMILES，
    因此要求 token 至少含一个非字母字符或长度 ≥ 4 且含环/键/括号符号之一。
    """
    if not isinstance(text, str) or not HAS_RDKIT:
        return set()
    out = set()
    for tok in _SMILES_TOKEN.findall(text):
        tok = tok.strip(".")
        if not tok or tok.isalpha() or tok.isdigit():
            continue
        if not any(c in tok for c in "()[]=#123456789@+-/\\"):
            continue
        mol = Chem.MolFromSmiles(tok)
        if mol is not None:
            out.add(Chem.MolToSmiles(mol))
    return out
