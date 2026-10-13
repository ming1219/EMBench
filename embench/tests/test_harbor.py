"""Harbor 集成的离线测试（不需要 Docker）：fixture 匹配、调用记录判分、导出产物。"""
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from embench.core.task import load_tasks
from embench.core.trajectory import Trajectory
from embench.core.verifier import verify
from embench.harbor.export import export_all
from embench.harbor.fixtures import FIXTURE_MISSING, answer_call, build_fixtures, trajectory_from_calls

TNT_A = "Cc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]"
TNT_B = "Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]"


def simulate(task, steps: list[dict], all_steps: list[dict]) -> list[dict]:
    """模拟 tool-server：逐个调用按 fixture 回放结果，并像服务端那样记录调用。"""
    oracle = [s.to_dict() for s in Trajectory.load(task.solution_path).steps]
    fixtures = build_fixtures(oracle, all_steps)
    return [{"seq": i, "tool": s["tool"], "args": s.get("args") or {},
             "result": answer_call(fixtures, s["tool"], s.get("args") or {})} for i, s in enumerate(steps)]


class TestFixtures(unittest.TestCase):
    def setUp(self):
        self.fx = build_fixtures(
            [{"tool": "calculateKJ", "args": {"smiles": TNT_A}, "result": "local"}],
            [{"tool": "calculateKJ", "args": {"smiles": TNT_A}, "result": "global"},
             {"tool": "calculateOxygenBalance", "args": {"smiles_list": [TNT_A]}, "result": "ob"},
             {"tool": "magusStatus", "args": {}, "result": "ok"}])

    def test_equivalent_smiles_and_json_string_lists_match(self):
        self.assertEqual(answer_call(self.fx, "calculateKJ", {"smiles": TNT_B}), "local")
        self.assertEqual(answer_call(self.fx, "calculateOxygenBalance", {"smiles_list": f'["{TNT_B}"]'}), "ob")

    def test_local_fixtures_win_and_extra_args_are_ignored(self):
        self.assertEqual(answer_call(self.fx, "calculateKJ", {"smiles": TNT_A, "density": 1.6}), "local")
        self.assertEqual(answer_call(self.fx, "magusStatus", {}), "ok")

    def test_missing_fixture_is_a_structured_error(self):
        out = json.loads(answer_call(self.fx, "calculateKJ", {"smiles": "CCO"}))
        self.assertEqual(out["error_type"], FIXTURE_MISSING)
        self.assertEqual(json.loads(answer_call(self.fx, "nameToSmiles", {"name": "TNT"}))["error_type"], FIXTURE_MISSING)

    def test_global_pool_prefers_real_results_over_synthetic_ones(self):
        from embench.harbor.export import global_fixture_steps
        fx = build_fixtures([], global_fixture_steps(load_tasks()))
        kj = json.loads(answer_call(fx, "calculateKJ", {"smiles": TNT_A}))
        self.assertEqual(kj["DetoD"], 6995)  # ob-kj/tnt 的真实计算结果，而不是设计题里的示意值

    def test_dedupe(self):
        fx = build_fixtures([], [{"tool": "t", "args": {"a": 1}, "result": "x"}] * 3)
        self.assertEqual(len(fx["global"]), 1)


class TestGradingFromServerLog(unittest.TestCase):
    """服务端调用记录 + 回答 → 判分。全部 74 道题的参考解经 fixture 回放后都必须通过。"""

    @classmethod
    def setUpClass(cls):
        cls.tasks = load_tasks()
        cls.all_steps = [s.to_dict() for t in cls.tasks for s in Trajectory.load(t.solution_path).steps]

    def test_every_reference_solution_passes_through_the_tool_server(self):
        for t in self.tasks:
            oracle = Trajectory.load(t.solution_path)
            calls = simulate(t, [s.to_dict() for s in oracle.steps], self.all_steps)
            for c, s in zip(calls, oracle.steps):  # 回放结果与参考轨迹一致
                self.assertEqual(c["result"], s.result, (t.id, s.tool))
            r = verify(t, trajectory_from_calls(calls, oracle.final_answer, t.id))
            self.assertTrue(r["pass"], (t.id, [c["detail"] for c in r["checks"] if c["passed"] is False]))

    def test_empty_log_fails_every_task(self):
        for t in self.tasks:
            self.assertFalse(verify(t, trajectory_from_calls([], "", t.id))["pass"], t.id)

    def test_answer_alone_without_tool_calls_fails(self):
        for t in self.tasks:
            oracle = Trajectory.load(t.solution_path)
            self.assertFalse(verify(t, trajectory_from_calls([], oracle.final_answer, t.id))["pass"], t.id)

    def test_grade_entry_without_log_fails(self):
        if importlib.util.find_spec("embench") is None:  # grade.py 在容器里以顶层包 embench 导入判分代码
            self.skipTest("embench 不是顶层包（内部版目录结构）")
        spec = importlib.util.spec_from_file_location(
            "embench_grade", Path(__file__).resolve().parents[1] / "harbor" / "runtime" / "grade.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        t = load_tasks(["ob-kj/tnt"])[0]
        r = mod.grade(t, None, "anything", "connection refused")
        self.assertFalse(r["pass"])
        self.assertIn("tool_call_log", r["missing_evidence"])


class TestExport(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())
        self.tasks = load_tasks(["ob-kj/tnt", "design-multi/q1-highenergy-lowfilter"])
        self.paths = export_all(self.tasks, load_tasks(), self.out, token="t0k3n")

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)

    def test_layout_is_self_contained(self):
        for p in self.paths:
            for rel in ("task.toml", "instruction.md", "environment/Dockerfile", "environment/docker-compose.yaml",
                        "environment/tool-server/Dockerfile", "environment/tool-server/tool_server.py",
                        "environment/tool-server/tools.json", "environment/tool-server/fixtures.json",
                        "environment/tool-server/embench/core/checks.py", "tests/test.sh", "tests/grade.py",
                        "tests/embench/core/verifier.py", "solution/solve.sh", "solution/solve.py",
                        "solution/trajectory.json"):
                self.assertTrue((p / rel).is_file(), (p.name, rel))

    def test_token_only_where_intended(self):
        for p in self.paths:
            hits = sorted(str(f.relative_to(p)).replace("\\", "/") for f in p.rglob("*")
                          if f.is_file() and "t0k3n" in f.read_text(encoding="utf-8", errors="ignore"))
            self.assertEqual(hits, ["environment/docker-compose.yaml", "task.toml"], p.name)

    def test_reference_answers_do_not_ship_with_the_grader(self):
        for p in self.paths:
            self.assertFalse(any((p / "tests").rglob("trajectory.json")), p.name)
            self.assertFalse(any((p / "tests").rglob("negatives")), p.name)

    def test_instruction_asks_for_answer_file_and_scripts_use_lf(self):
        for p in self.paths:
            self.assertIn("/app/answer.md", (p / "instruction.md").read_text(encoding="utf-8"))
            for rel in ("tests/test.sh", "solution/solve.sh"):
                self.assertNotIn(b"\r\n", (p / rel).read_bytes(), rel)


if __name__ == "__main__":
    unittest.main()
