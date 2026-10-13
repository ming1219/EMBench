"""公开工具的复现测试：对种子题参考轨迹里记录的每一次调用，公开实现算出的结果必须与记录一致。

PubChem 在线查询默认跳过；设置环境变量 EMBENCH_NETWORK_TESTS=1 后才会联网测试。
"""
import json
import os
import unittest
from pathlib import Path

from embench.core.normalize import canonical_smiles
from embench.core.task import load_tasks
from embench.core.trajectory import Trajectory
from embench.tools import OPEN_TOOLS, call_open_tool


def recorded_calls(tool: str):
    """所有种子题参考轨迹中，某个工具的 (task_id, args, 记录的结果)。"""
    out = []
    for t in load_tasks():
        for s in Trajectory.load(t.solution_path).steps:
            if s.tool == tool and s.has_result:
                out.append((t.id, s.args, json.loads(s.result)))
    return out


def run(tool, args):
    return json.loads(call_open_tool(tool, args))


class TestReproducesRecordedResults(unittest.TestCase):
    def test_validate_smiles(self):
        calls = recorded_calls("validateSmiles")
        self.assertGreater(len(calls), 0)
        for tid, args, rec in calls:
            got = run("validateSmiles", args)
            for k in ("valid", "formula", "num_atoms", "num_bonds"):
                self.assertEqual(got[k], rec[k], (tid, k))

    def test_smiles_to_3d(self):
        calls = recorded_calls("smilesTo3D")
        self.assertGreater(len(calls), 0)
        for tid, args, rec in calls:
            got = run("smilesTo3D", args)
            for k in ("formula", "num_atoms", "bond_count"):
                self.assertEqual(got[k], rec[k], (tid, k))
            self.assertEqual(len(got["atoms"]), got["num_atoms"], tid)

    def test_oxygen_balance(self):
        calls = recorded_calls("calculateOxygenBalance")
        self.assertGreater(len(calls), 0)
        for tid, args, rec in calls:
            got = run("calculateOxygenBalance", args)
            self.assertEqual(len(got), len(rec), tid)
            for g, r in zip(got, rec):
                self.assertEqual(g["formula"], r["formula"], tid)
                for k in ("OB_CO", "OB_CO2"):
                    self.assertAlmostEqual(g[k], r[k], places=3, msg=(tid, k))

    def test_name_to_smiles_builtin(self):
        calls = [c for c in recorded_calls("nameToSmiles") if c[2].get("source") == "builtin"]
        self.assertGreater(len(calls), 0)
        for tid, args, rec in calls:
            got = run("nameToSmiles", args)
            self.assertEqual(got["source"], "builtin", tid)
            self.assertEqual(canonical_smiles(got["smiles"]), canonical_smiles(rec["smiles"]), tid)

    @unittest.skipUnless(os.environ.get("EMBENCH_NETWORK_TESTS") == "1", "PubChem 联网测试：设置 EMBENCH_NETWORK_TESTS=1 启用")
    def test_name_to_smiles_pubchem(self):
        calls = [c for c in recorded_calls("nameToSmiles") if c[2].get("source") != "builtin"]
        for tid, args, rec in calls:
            got = run("nameToSmiles", args)
            self.assertEqual(canonical_smiles(got.get("smiles", "")), canonical_smiles(rec["smiles"]), tid)

    def test_draw_molecule(self):
        for tid, args, rec in recorded_calls("drawMolecule")[:3]:
            got = run("drawMolecule", args)
            self.assertEqual((got["width"], got["height"]), (rec["width"], rec["height"]), tid)
            self.assertTrue(Path(got["image_path"]).is_file(), tid)


class TestErrors(unittest.TestCase):
    def test_invalid_input_returns_structured_error(self):
        self.assertFalse(run("validateSmiles", {"smiles": "not-a-smiles"})["valid"])
        self.assertIn("error_type", run("smilesTo3D", {"smiles": "xx"}))
        self.assertIn("error_type", run("drawMolecule", {"smiles": ""}))
        self.assertIn("error_type", run("calculateOxygenBalance", {"smiles_list": []}))

    def test_open_tools_list(self):
        self.assertEqual(OPEN_TOOLS, {"nameToSmiles", "validateSmiles", "drawMolecule", "smilesTo3D",
                                      "calculateOxygenBalance"})

    def test_json_string_list_argument(self):
        got = run("calculateOxygenBalance", {"smiles_list": '["Cc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]"]'})
        self.assertAlmostEqual(got[0]["OB_CO2"], -73.9658, places=3)


if __name__ == "__main__":
    unittest.main()
