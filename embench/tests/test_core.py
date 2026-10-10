"""核心层单测：归一化 / 检查项 / 指标。纯离线，不依赖 LLM。

    python -m unittest discover -s embench/tests -t .
"""
import unittest
from types import SimpleNamespace

from embench.core import normalize as N
from embench.core.checks import run_check
from embench.core.metrics import pass_at_k, pass_hat_k, wilson
from embench.core.trajectory import Step, Trajectory

TNT_A = "Cc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]"
TNT_B = "Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]"  # 同分子不同写法
TASK = SimpleNamespace(dir=None)


def traj(*steps, answer=""):
    return Trajectory(steps=[Step(t, a, r, r is not None) for t, a, r in steps], final_answer=answer)


def check(t, **spec):
    return run_check(t, spec, TASK)


class TestNormalize(unittest.TestCase):
    @unittest.skipUnless(N.HAS_RDKIT, "需要 rdkit")
    def test_smiles_canonical(self):
        self.assertEqual(N.canonical_smiles(TNT_A), N.canonical_smiles(TNT_B))
        self.assertTrue(N.values_equal(TNT_A, TNT_B, "smiles"))

    def test_coerce_list(self):
        self.assertEqual(N.coerce_list('["a","b"]'), ["a", "b"])
        self.assertEqual(N.coerce_list("a, b"), ["a", "b"])
        self.assertEqual(N.coerce_list(None), [])
        self.assertEqual(N.coerce_list("x"), ["x"])

    def test_smiles_list_as_json_string(self):
        # LLM 常把数组参数传成 JSON 字符串 —— 不应判错
        self.assertTrue(N.values_equal(f'["{TNT_B}"]', [TNT_A], "smiles_set"))

    def test_alias(self):
        self.assertTrue(N.values_equal("CL20", "CL-20", "alias"))
        self.assertFalse(N.values_equal("TNT", "trinitrotoluene", "alias"))

    def test_properties_set(self):
        self.assertTrue(N.values_equal(["DetoD", "Density"], '["density","detod"]', "str_set"))

    def test_numbers(self):
        self.assertTrue(N.values_equal(3, "3", None))
        self.assertTrue(N.values_equal(1.0001, 1.0, None))
        self.assertFalse(N.values_equal(1.1, 1.0, None))

    @unittest.skipUnless(N.HAS_RDKIT, "需要 rdkit")
    def test_extract_smiles_from_chinese(self):
        text = f"**结论**：TNT 的 SMILES 为 `{TNT_B}`，来源内置库。"
        self.assertIn(N.canonical_smiles(TNT_A), N.extract_smiles(text))
        self.assertEqual(N.extract_smiles("TNT 与 SMILES 都只是单词"), set())


class TestToolCallsMatch(unittest.TestCase):
    CALLS = [{"tool": "calculateOxygenBalance", "args": {"smiles_list": [TNT_A]}},
             {"tool": "calculateKJ", "args": {"smiles": TNT_A}}]

    def test_exact(self):
        t = traj(("calculateOxygenBalance", {"smiles_list": [TNT_B]}, None), ("calculateKJ", {"smiles": TNT_B}, None))
        r = check(t, type="tool_calls_match", calls=self.CALLS)
        self.assertTrue(r.passed)
        self.assertTrue(r.data["exact"])

    def test_redundant_calls_allowed(self):
        t = traj(("validateSmiles", {"smiles": TNT_A}, None),
                 ("calculateOxygenBalance", {"smiles_list": f'["{TNT_A}"]'}, None),
                 ("calculateKJ", {"smiles": TNT_A}, None))
        r = check(t, type="tool_calls_match", calls=self.CALLS)
        self.assertTrue(r.passed)
        self.assertEqual(r.data["redundancy"], 1)
        r = check(t, type="tool_calls_match", calls=self.CALLS, allow_extra=False)
        self.assertFalse(r.passed)

    def test_wrong_order(self):
        t = traj(("calculateKJ", {"smiles": TNT_A}, None), ("calculateOxygenBalance", {"smiles_list": [TNT_A]}, None))
        self.assertFalse(check(t, type="tool_calls_match", calls=self.CALLS).passed)
        self.assertTrue(check(t, type="tool_calls_match", calls=self.CALLS, ordered=False).passed)

    def test_wrong_args_scores_l1_not_l2(self):
        t = traj(("calculateOxygenBalance", {"smiles_list": ["CCO"]}, None), ("calculateKJ", {"smiles": "CCO"}, None))
        r = check(t, type="tool_calls_match", calls=self.CALLS)
        self.assertEqual(r.scores, {"l1_tool": 1.0, "l2_args": 0.0})
        self.assertFalse(r.passed)

    def test_kj_direct_mode_rejected(self):
        t = traj(("calculateOxygenBalance", {"smiles_list": [TNT_A]}, None), ("calculateKJ", {"density": 1.6}, None))
        self.assertFalse(check(t, type="tool_calls_match", calls=self.CALLS).passed)


class TestArgConstraints(unittest.TestCase):
    def test_ops_and_default(self):
        t = traj(("generateAndPredict", {"num": "5", "properties": '["Density"]'}, None))
        r = check(t, type="arg_constraints", tool="generateAndPredict",
                  constraints={"num": {"eq": 5}, "use_em": {"eq": True, "default": True},
                               "properties": {"set_eq": ["Density"]}, "skip_prediction": {"absent": True}})
        self.assertTrue(r.passed, r.detail)

    def test_missing_without_default_fails(self):
        t = traj(("predictRetrosynthesis", {"smiles": "C", "max_routes": 3}, None))
        r = check(t, type="arg_constraints", tool="predictRetrosynthesis", which="all",
                  constraints={"max_routes": {"le": 3}, "time_limit": {"le": 30}})
        self.assertFalse(r.passed)
        self.assertAlmostEqual(r.scores["l2_args"], 0.5)

    def test_sum(self):
        t = traj(("generateAndPredict", {"num": 2}, None), ("generateAndPredict", {"num": 2}, None))
        spec = dict(type="arg_constraints", tool="generateAndPredict", constraints={"num": {"eq": 4}})
        self.assertFalse(check(t, **spec).passed)
        self.assertTrue(check(t, which="sum", **spec).passed)

    def test_not_called(self):
        r = check(traj(), type="arg_constraints", tool="x", constraints={"a": {"eq": 1}})
        self.assertFalse(r.passed)


class TestResultChecks(unittest.TestCase):
    def test_values_with_list_result_and_zero(self):
        res = f'[{{"smiles": "{TNT_B}", "OB_CO": 0.0, "OB_CO2": -73.97}}]'
        t = traj(("calculateOxygenBalance", {}, res))
        r = check(t, type="tool_result_values", tool="calculateOxygenBalance", match_smiles=TNT_A,
                  expected={"OB_CO": 0.0, "OB_CO2": -73.9658}, rel_tol=0.01, abs_tol=0.01)
        self.assertTrue(r.passed, r.detail)

    def test_skipped_without_results(self):
        t = traj(("calculateKJ", {"smiles": "C"}, None))
        r = check(t, type="tool_result_values", tool="calculateKJ", expected={"DetoD": 1})
        self.assertIsNone(r.passed)

    def test_coverage_and_consistency(self):
        gen = '{"generated_molecules": ["CCO", "CCN"]}'
        t = traj(("generateAndPredict", {}, gen),
                 ("calculateKJ", {"smiles": "OCC"}, "{}"), ("calculateKJ", {"smiles": "CCN"}, "{}"),
                 ("predictRetrosynthesis", {"smiles": "CCO"}, "{}"), ("predictRetrosynthesis", {"smiles": "c1ccccc1"}, "{}"))
        r = check(t, type="candidate_coverage", downstream={"calculateKJ": "smiles", "predictRetrosynthesis": "smiles"},
                  gate=["coverage"])
        self.assertAlmostEqual(r.scores["coverage"], 0.5)   # CCN 没做逆合成
        self.assertAlmostEqual(r.scores["consistency"], 0.75)  # 苯是编造的
        self.assertFalse(r.passed)

    def test_answer_checks(self):
        t = traj(answer=f"推荐候选 {TNT_B}，爆速最高。")
        self.assertTrue(check(t, type="answer_contains", any_of=["推荐"], min_length=5).passed)
        if N.HAS_RDKIT:
            self.assertTrue(check(t, type="answer_smiles", expected=[TNT_A]).passed)

    def test_budget_is_soft(self):
        t = traj(("a", {}, None), ("b", {}, None), ("c", {}, None), ("d", {}, None))
        r = check(t, type="call_budget", max_calls=2)
        self.assertFalse(r.required)
        self.assertAlmostEqual(r.scores["efficiency"], 0.5)

    def test_check_crash_is_contained(self):
        t = traj(("generateAndPredict", {}, '{"generated_molecules": 5}'))  # 非法结构
        r = check(t, type="candidate_coverage", downstream={"calculateKJ": "smiles"})
        self.assertFalse(r.passed)


class TestLegacyTrajectory(unittest.TestCase):
    def test_pairs_results_by_order(self):
        wf = {"tool_calls": [{"calculateKJ": {"smiles": "A"}}, {"calculateKJ": {"smiles": "B"}}, {"x": {}}],
              "tool_results": {"calculateKJ": ["ra", "rb"]}, "result": "done"}
        t = Trajectory.from_legacy(wf)
        self.assertEqual([s.result for s in t.steps], ["ra", "rb", None])
        self.assertEqual([s.has_result for s in t.steps], [True, True, False])

    def test_roundtrip(self):
        t = traj(("a", {"x": 1}, "r"), answer="ok")
        self.assertEqual(Trajectory.from_dict(t.to_dict()).to_dict(), t.to_dict())


class TestMetrics(unittest.TestCase):
    def test_pass_at_k(self):
        self.assertEqual(pass_at_k(3, 0, 3), 0.0)
        self.assertEqual(pass_at_k(3, 1, 3), 1.0)
        self.assertAlmostEqual(pass_at_k(4, 2, 2), 1 - 1 / 6)
        self.assertAlmostEqual(pass_hat_k(4, 2, 2), 1 / 6)
        self.assertEqual(pass_hat_k(3, 3, 3), 1.0)

    def test_wilson(self):
        lo, hi = wilson(10, 10)
        self.assertLess(lo, 1.0)
        self.assertEqual(hi, 1.0)
        self.assertEqual(wilson(0, 0), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
