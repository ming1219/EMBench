"""
公开工具的实现：只依赖 RDKit 与 Python 标准库（PubChem 查询用 urllib）。

返回格式与种子题参考轨迹里记录的结果一致（字段名、单位、数值精度），
tests/test_open_tools.py 逐题核对「公开实现算出的结果 == 当初记录的结果」。
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from ..core.normalize import coerce_list

RDLogger.DisableLog("rdApp.*")

OPEN_TOOLS = {"nameToSmiles", "validateSmiles", "drawMolecule", "smilesTo3D", "calculateOxygenBalance"}

# 常用分子名（离线可用）。键为归一化后的名字：小写、去掉空格 / 连字符 / 下划线
_BUILTIN = {
    "tnt": "Cc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
    "trinitrotoluene": "Cc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
    "246trinitrotoluene": "Cc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
    "rdx": "O=[N+]([O-])N1CN([N+](=O)[O-])CN([N+](=O)[O-])C1",
    "hexogen": "O=[N+]([O-])N1CN([N+](=O)[O-])CN([N+](=O)[O-])C1",
    "hmx": "O=[N+]([O-])N1CN([N+](=O)[O-])CN([N+](=O)[O-])CN([N+](=O)[O-])C1",
    "octogen": "O=[N+]([O-])N1CN([N+](=O)[O-])CN([N+](=O)[O-])CN([N+](=O)[O-])C1",
    "cl20": "O=[N+]([O-])N1C2C3N([N+](=O)[O-])C1C1N([N+](=O)[O-])C(C(N1[N+](=O)[O-])N3[N+](=O)[O-])N2[N+](=O)[O-]",
    "hniw": "O=[N+]([O-])N1C2C3N([N+](=O)[O-])C1C1N([N+](=O)[O-])C(C(N1[N+](=O)[O-])N3[N+](=O)[O-])N2[N+](=O)[O-]",
    "tatb": "Nc1c([N+](=O)[O-])c(N)c([N+](=O)[O-])c(N)c1[N+](=O)[O-]",
    "petn": "O=[N+]([O-])OCC(CO[N+](=O)[O-])(CO[N+](=O)[O-])CO[N+](=O)[O-]",
    "ng": "O=[N+]([O-])OCC(O[N+](=O)[O-])CO[N+](=O)[O-]",
    "nitroglycerin": "O=[N+]([O-])OCC(O[N+](=O)[O-])CO[N+](=O)[O-]",
    "nitroglycerine": "O=[N+]([O-])OCC(O[N+](=O)[O-])CO[N+](=O)[O-]",
}
_PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/{}/JSON"
# PubChem 的 SMILES 字段名在不同时期不同，按顺序取第一个非空的
_SMILES_FIELDS = ("SMILES", "IsomericSMILES", "CanonicalSMILES", "ConnectivitySMILES")


def _norm_name(name: str) -> str:
    return re.sub(r"[\s\-_,]", "", str(name)).lower()


def _mol(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip())


def _error(message: str, error_type: str) -> dict:
    return {"error": message, "error_type": error_type}


# ----------------------------------------------------------------
#  工具实现
# ----------------------------------------------------------------

def name_to_smiles(name: str, timeout: float = 15.0) -> dict:
    key = _norm_name(name)
    if key in _BUILTIN:
        return {"smiles": _BUILTIN[key], "source": "builtin", "name": name}
    url = _PUBCHEM.format(urllib.parse.quote(str(name).strip()), ",".join(_SMILES_FIELDS))
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            props = json.loads(resp.read().decode("utf-8"))["PropertyTable"]["Properties"][0]
    except Exception as e:
        return _error(f"PubChem 未找到「{name}」或无法访问：{type(e).__name__}", "not_found")
    smiles = next((props[f] for f in _SMILES_FIELDS if props.get(f)), None)
    mol = _mol(smiles)
    if mol is None:
        return _error(f"PubChem 没有返回「{name}」的有效 SMILES", "not_found")
    return {"smiles": Chem.MolToSmiles(mol), "source": "pubchem", "name": name, "cid": props.get("CID")}


def validate_smiles(smiles: str) -> dict:
    mol = _mol(smiles)
    if mol is None:
        return {"valid": False, "error": "RDKit 无法解析该 SMILES"}
    return {"valid": True, "formula": rdMolDescriptors.CalcMolFormula(mol),
            "num_atoms": mol.GetNumAtoms(), "num_bonds": mol.GetNumBonds()}


def draw_molecule(smiles: str, width: int = 400, height: int = 400) -> dict:
    from rdkit.Chem import Draw  # 用到时再导入：Draw 依赖 X11 绘图库，精简镜像里缺失时不影响其他工具

    mol = _mol(smiles)
    if mol is None:
        return _error("RDKit 无法解析该 SMILES", "invalid_smiles")
    width, height = int(width or 400), int(height or 400)
    out_dir = Path(os.environ.get("EMBENCH_IMAGE_DIR", Path(tempfile.gettempdir()) / "embench_images"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"mol_{abs(hash(Chem.MolToSmiles(mol))) % 10**10}_{width}x{height}.png"
    Draw.MolToFile(mol, str(path), size=(width, height))
    return {"smiles": smiles, "message": "图片已保存", "width": width, "height": height, "image_path": str(path)}


def smiles_to_3d(smiles: str, seed: int = 42) -> dict:
    mol = _mol(smiles)
    if mol is None:
        return _error("RDKit 无法解析该 SMILES", "invalid_smiles")
    mh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mh, params) != 0:
        return _error("无法生成 3D 构象", "embed_failed")
    try:
        AllChem.MMFFOptimizeMolecule(mh)
    except Exception:
        pass  # 力场不支持时保留 ETKDG 构象
    conf = mh.GetConformer()
    atoms = [{"symbol": a.GetSymbol(), "x": round(p.x, 4), "y": round(p.y, 4), "z": round(p.z, 4)}
             for a, p in ((a, conf.GetAtomPosition(a.GetIdx())) for a in mh.GetAtoms())]
    return {"formula": rdMolDescriptors.CalcMolFormula(mh), "num_atoms": mh.GetNumAtoms(),
            "bond_count": mh.GetNumBonds(), "atoms": atoms}


def oxygen_balance(smiles_list) -> list | dict:
    """OB_CO2 = -1600 (2C + H/2 - O) / M；OB_CO = -1600 (C + H/2 - O) / M（单位 %，保留 4 位小数）。"""
    out = []
    for s in coerce_list(smiles_list):
        mol = _mol(str(s))
        if mol is None:
            out.append({"smiles": s, "error": "RDKit 无法解析该 SMILES"})
            continue
        mh = Chem.AddHs(mol)
        count = {"C": 0, "H": 0, "O": 0}
        for a in mh.GetAtoms():
            if a.GetSymbol() in count:
                count[a.GetSymbol()] += 1
        mw = Descriptors.MolWt(mol)
        c, h, o = count["C"], count["H"], count["O"]
        out.append({"smiles": s, "formula": rdMolDescriptors.CalcMolFormula(mol),
                    "OB_CO": round(-1600 * (c + h / 2 - o) / mw, 4),
                    "OB_CO2": round(-1600 * (2 * c + h / 2 - o) / mw, 4)})
    return out if out else _error("smiles_list 为空", "invalid_input")


_IMPL = {
    "nameToSmiles": lambda a: name_to_smiles(a.get("name", "")),
    "validateSmiles": lambda a: validate_smiles(a.get("smiles", "")),
    "drawMolecule": lambda a: draw_molecule(a.get("smiles", ""), a.get("width") or 400, a.get("height") or 400),
    "smilesTo3D": lambda a: smiles_to_3d(a.get("smiles", "")),
    "calculateOxygenBalance": lambda a: oxygen_balance(a.get("smiles_list")),
}


def call_open_tool(tool: str, args: dict) -> str:
    """调用公开工具，返回 JSON 字符串（与工具服务的返回格式一致）。"""
    try:
        result = _IMPL[tool](args or {})
    except Exception as e:
        result = _error(f"{type(e).__name__}: {e}", "tool_error")
    return json.dumps(result, ensure_ascii=False)
