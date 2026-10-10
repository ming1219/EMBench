"""题库级测试：真实 74 道题全部通过自检 + 自检器能抓出坏题 + oracle/nop 端到端 job。"""
import asyncio
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from embench.agents.simple import NopAgent, OracleAgent, ReplayAgent
from embench.core.checker import check_registry, check_task
from embench.core.runner import rescore_job, run_job
from embench.core.task import TASKS_DIR, load_tasks

SAMPLE = "ob-kj/tnt"


class TestRealSuite(unittest.TestCase):
    def test_every_task_passes_self_check(self):
        tasks = load_tasks()
        self.assertGreater(len(tasks), 0)
        bad = {t.id: rep.errors for t in tasks if not (rep := check_task(t.dir)).ok}
        self.assertEqual(bad, {})

    def test_registry_up_to_date(self):
        self.assertEqual(check_registry(), [])


class TestCheckerCatchesBrokenTasks(unittest.TestCase):
    """变异测试：故意把一道好题改坏，自检器必须报错。"""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        shutil.copytree(TASKS_DIR / SAMPLE, self.root / SAMPLE)
        self.dir = self.root / SAMPLE
        self.toml = (self.dir / "task.toml").read_text(encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def errors(self):
        return check_task(self.dir, self.root).errors

    def _edit(self, old, new):
        assert old in self.toml, old
        (self.dir / "task.toml").write_text(self.toml.replace(old, new, 1), encoding="utf-8")

    def test_baseline_ok(self):
        self.assertEqual(self.errors(), [])

    def test_param_typo(self):
        self._edit("ordered = true", "orderd = true")
        self.assertTrue(any("未知参数" in e for e in self.errors()))

    def test_unknown_check_type(self):
        self._edit('type = "call_budget"', 'type = "call_budgett"')
        self.assertTrue(any("未知检查类型" in e for e in self.errors()))

    def test_wrong_expected_value_breaks_oracle(self):
        self._edit("DetoD = 6995", "DetoD = 9999")
        self.assertTrue(any("oracle 未通过" in e for e in self.errors()))

    def test_too_lenient_verifier(self):
        # 把轨迹检查改成只要求第一步 → 截断 oracle 仍能通过 → 必须报「检查项过松」
        start = self.toml.index("[[verifier.checks]]")
        lenient = (self.toml[:start] +
                   '[[verifier.checks]]\nid = "only_ob"\ntype = "tools_used"\nall_of = ["calculateOxygenBalance"]\n')
        (self.dir / "task.toml").write_text(lenient, encoding="utf-8")
        self.assertTrue(any("截断" in e for e in self.errors()))

    def test_negative_that_passes_is_reported(self):
        oracle = (self.dir / "solution" / "trajectory.json").read_text(encoding="utf-8")
        (self.dir / "solution" / "negatives" / "fake.json").write_text(oracle, encoding="utf-8")
        self.assertTrue(any("反例 fake.json" in e for e in self.errors()))

    def test_id_mismatch(self):
        self._edit('id = "ob-kj/tnt"', 'id = "ob-kj/rdx"')
        self.assertTrue(any("加载失败" in e for e in self.errors()))


class TestEndToEndJob(unittest.TestCase):
    def setUp(self):
        self.jobs = Path(tempfile.mkdtemp())
        self.tasks = load_tasks(["ob-kj/*", "design-multi/*"])

    def tearDown(self):
        shutil.rmtree(self.jobs, ignore_errors=True)

    def run_agent(self, agent, name, k=1):
        return asyncio.run(run_job(agent, self.tasks, self.jobs / name, n_trials=k, log=lambda *_: None))

    def test_oracle_all_pass_nop_all_fail(self):
        r = self.run_agent(OracleAgent(), "oracle", k=2)
        self.assertEqual(r["overall"]["pass@1"], 1.0)
        self.assertEqual(r["overall"]["pass^2"], 1.0)
        self.assertTrue((self.jobs / "oracle" / "report.md").is_file())
        r = self.run_agent(NopAgent(), "nop")
        self.assertEqual(r["overall"]["pass@1"], 0.0)

    def test_replay_own_trajectories(self):
        src = self.jobs / "my_agent_outputs"
        (src / "ob-kj").mkdir(parents=True)
        shutil.copy(TASKS_DIR / SAMPLE / "solution" / "trajectory.json", src / f"{SAMPLE}.json")
        r = self.run_agent(ReplayAgent(str(src)), "replay")
        per = r["per_task"]
        self.assertTrue(per[f"{SAMPLE}#0"]["pass"])
        self.assertEqual(sum(v["pass"] for v in per.values()), 1)  # 其余题没有文件 → 判失败
        self.assertEqual(r["overall"]["error_trials"], len(self.tasks) - 1)

    def test_rescore_and_resume(self):
        self.run_agent(OracleAgent(), "j")
        reward = next((self.jobs / "j" / "trials").rglob("reward.json"))
        reward.write_text(json.dumps({**json.loads(reward.read_text(encoding="utf-8")), "pass": False}), encoding="utf-8")
        # 续跑：已有 reward.json 的 trial 不重跑
        r = self.run_agent(OracleAgent(), "j")
        self.assertLess(r["overall"]["pass@1"], 1.0)
        # 重判：按当前任务定义重新跑 verifier，恢复正确结果
        r = rescore_job(self.jobs / "j", log=lambda *_: None)
        self.assertEqual(r["overall"]["pass@1"], 1.0)


if __name__ == "__main__":
    unittest.main()
