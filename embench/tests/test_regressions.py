"""回归测试：外部评审发现的框架问题，每个问题一个反例，防止再犯。

1. 复用同一 job 目录换 agent，旧成绩被贴上新 agent 的标签
2. 缺少证据（轨迹没记录工具结果）的必查项被跳过，整题仍算通过
3. 同一份轨迹重复回放，被当成独立样本，置信区间虚假收窄
4. 关键词判推荐：「我不推荐任何候选」也含「推荐」
5. 只「提到」生成过的分子就算推荐：「候选包括 A，建议暂缓」「候选包括 A，推荐集外的 B」
6. 续跑子集时悄悄刷新其他题的版本记录，按旧定义判出的成绩混进报告
7. 用 null 冒充证据：候选列表塞 null、性质值全是 null，却数值满分且不标证据不完整
"""
import asyncio
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from embench.agents.simple import NopAgent, OracleAgent, ReplayAgent
from embench.core.metrics import aggregate, wilson
from embench.core.runner import JobError, rescore_job, run_job, trial_dir
from embench.core.task import TASKS_DIR, load_task, load_tasks
from embench.core.trajectory import Trajectory
from embench.core.verifier import verify

QUIET = {"log": lambda *_: None}
REFUSAL = "我不推荐任何候选，也没有完成比较，这个问题我无法回答。"
OUTSIDE = "O=[N+]([O-])c1nn[nH]n1"  # 不在任何设计题生成结果中的真实分子


def run(agent, tasks, job_dir, **kw):
    return asyncio.run(run_job(agent, tasks, job_dir, **QUIET, **kw))


class _TmpDir(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestJobIdentity(_TmpDir):
    def setUp(self):
        super().setUp()
        self.tasks = load_tasks(["name2smiles/tnt"])
        self.job = self.tmp / "job"

    def test_same_job_different_agent_is_rejected(self):
        self.assertEqual(run(OracleAgent(), self.tasks, self.job)["overall"]["pass@1"], 1.0)
        with self.assertRaises(JobError):
            run(NopAgent(), self.tasks, self.job)
        # 被拒绝后 job 记录不被改写，仍属于 oracle
        cfg = json.loads((self.job / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["agent"]["name"], "oracle")

    def test_force_starts_over(self):
        run(OracleAgent(), self.tasks, self.job)
        r = run(NopAgent(), self.tasks, self.job, force=True)
        self.assertEqual(r["overall"]["pass@1"], 0.0)
        cfg = json.loads((self.job / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["agent"]["name"], "nop")

    def test_same_agent_resumes(self):
        run(OracleAgent(), self.tasks, self.job)
        self.assertEqual(run(OracleAgent(), self.tasks, self.job)["overall"]["pass@1"], 1.0)

    def test_trial_from_other_agent_is_rejected_at_aggregation(self):
        run(OracleAgent(), self.tasks, self.job)
        reward_path = trial_dir(self.job, "name2smiles/tnt", 0) / "reward.json"
        reward = json.loads(reward_path.read_text(encoding="utf-8"))
        reward["agent"] = {"name": "someone-else"}
        reward_path.write_text(json.dumps(reward), encoding="utf-8")
        with self.assertRaises(JobError):
            run(OracleAgent(), self.tasks, self.job)

    def test_results_without_config_are_rejected(self):
        run(OracleAgent(), self.tasks, self.job)
        (self.job / "config.json").unlink()
        with self.assertRaises(JobError):
            run(NopAgent(), self.tasks, self.job)

    def test_changed_task_definition_blocks_resume(self):
        root = self.tmp / "tasks"
        shutil.copytree(TASKS_DIR / "name2smiles" / "tnt", root / "name2smiles" / "tnt")
        task = load_task(root / "name2smiles" / "tnt", root)
        run(OracleAgent(), [task], self.job)
        toml = task.dir / "task.toml"
        toml.write_text(toml.read_text(encoding="utf-8").replace("timeout_sec = 180", "timeout_sec = 181"),
                        encoding="utf-8")
        with self.assertRaises(JobError):
            run(OracleAgent(), [load_task(task.dir, root)], self.job)


def _strip_results(traj: Trajectory) -> Trajectory:
    out = Trajectory.from_dict(traj.to_dict())
    for s in out.steps:
        s.result, s.has_result = None, False
    return out


class TestMissingEvidence(_TmpDir):
    def test_stripped_results_never_pass_when_evidence_is_needed(self):
        for t in load_tasks():
            r = verify(t, _strip_results(Trajectory.load(t.solution_path)))
            if r["incomplete"]:
                self.assertFalse(r["pass"], f"{t.id} 缺证据 {r['missing_evidence']} 仍判通过")
            else:  # 不依赖工具结果的题（如 name2smiles）不受影响
                self.assertTrue(r["pass"], t.id)

    def test_allow_incomplete_is_explicit_opt_in(self):
        root = self.tmp / "tasks"
        shutil.copytree(TASKS_DIR / "ob-kj" / "tnt", root / "ob-kj" / "tnt")
        toml = root / "ob-kj" / "tnt" / "task.toml"
        s = toml.read_text(encoding="utf-8")
        i = s.index("[[verifier.checks]]")
        toml.write_text(s[:i] + "[verifier]\nallow_incomplete = true\n\n" + s[i:], encoding="utf-8")
        task = load_task(root / "ob-kj" / "tnt", root)
        r = verify(task, _strip_results(Trajectory.load(task.solution_path)))
        self.assertTrue(r["pass"])
        self.assertTrue(r["incomplete"])


class TestStatistics(_TmpDir):
    def test_deterministic_agents_refuse_repeated_trials(self):
        tasks = load_tasks(["name2smiles/tnt"])
        for agent in (OracleAgent(), NopAgent()):
            with self.assertRaises(JobError):
                run(agent, tasks, self.tmp / agent.name, n_trials=2)

    def test_ci_counts_tasks_not_trials(self):
        def trials(n_rep):
            return [{"task_id": f"t{i}", "family": "f", "trial": k,
                     "reward": {"pass": True, "dimensions": {}, "stats": {"n_tool_calls": 1, "error": None}}}
                    for i in range(2) for k in range(n_rep)]
        once, hundred = aggregate(trials(1))["overall"], aggregate(trials(100))["overall"]
        self.assertEqual(once["wilson_95"], hundred["wilson_95"])
        self.assertEqual(once["wilson_95"], list(wilson(2, 2)))

    def test_single_replay_file_counts_once(self):
        tasks = load_tasks(["ob-kj/tnt", "name2smiles/tnt"])
        src = self.tmp / "src"
        for t in tasks:
            (src / t.family).mkdir(parents=True, exist_ok=True)
            shutil.copy(t.solution_path, src / f"{t.id}.json")
        r1 = run(ReplayAgent(str(src)), tasks, self.tmp / "k1")["overall"]
        r3 = run(ReplayAgent(str(src)), tasks, self.tmp / "k3", n_trials=3)["overall"]
        self.assertEqual(r1["pass@1"], 1.0)
        self.assertAlmostEqual(r3["pass@1"], 1 / 3)  # 第 1、2 次试验没有独立记录 → 判失败，而不是复用
        self.assertEqual(r3["error_trials"], 4)
        self.assertLessEqual(r3["wilson_95"][1] - r3["wilson_95"][0], 1.0)
        self.assertEqual(r1["wilson_95"], list(wilson(2, 2)))

    def test_duplicate_trial_files_are_flagged(self):
        t = load_tasks(["ob-kj/tnt"])[0]
        d = self.tmp / "src" / t.id
        d.mkdir(parents=True)
        for k in range(3):
            shutil.copy(t.solution_path, d / f"trial_{k}.json")
        r = run(ReplayAgent(str(self.tmp / "src")), [t], self.tmp / "dup", n_trials=3)["overall"]
        self.assertEqual(r["duplicate_trials"], 2)
        self.assertEqual(r["wilson_95"], list(wilson(1, 1)))


class TestRecommendationAnswer(unittest.TestCase):
    def setUp(self):
        self.design = load_tasks(["design-*"])
        self.assertEqual(len(self.design), 6)

    def _with_answer(self, task, answer):
        traj = Trajectory.load(task.solution_path)
        traj.final_answer = answer
        return verify(task, traj)

    def test_refusal_fails(self):
        for t in self.design:
            r = self._with_answer(t, REFUSAL)
            self.assertFalse(r["pass"], t.id)
            self.assertLess(r["dimensions"]["answer"], 1.0, t.id)

    def test_recommending_a_molecule_never_generated_fails(self):
        for t in self.design:
            self.assertFalse(self._with_answer(t, "综合比较后推荐候选 CCO，爆速最高、合成最简单。")["pass"], t.id)

    def test_vague_recommendation_without_candidate_fails(self):
        for t in self.design:
            self.assertFalse(self._with_answer(t, "综合来看，推荐第一个候选，它的爆速最高。")["pass"], t.id)

    def test_recommend_one_and_reject_another_still_passes(self):
        for t in self.design:
            gen = self._generated(t)
            answer = f"推荐候选 {gen[0]}，爆速最高；不推荐 {gen[1]}，合成路线过长。"
            self.assertTrue(self._with_answer(t, answer)["pass"], t.id)

    # ---- 第二轮评审：只「提到」生成过的分子不等于推荐了它 ----

    def _generated(self, task):
        return json.loads(Trajectory.load(task.solution_path).steps[0].result)["generated_molecules"]

    def test_listing_a_candidate_then_deferring_fails(self):
        for t in self.design:
            a = self._generated(t)[0]
            r = self._with_answer(t, f"本次生成的候选包括 {a}。建议暂缓选择，现有证据不足以作出结论。")
            self.assertFalse(r["pass"], t.id)
            self.assertLess(r["dimensions"]["answer"], 1.0, t.id)

    def test_listing_a_candidate_then_recommending_an_outside_molecule_fails(self):
        for t in self.design:
            gen = self._generated(t)
            self.assertNotIn(OUTSIDE, gen)
            r = self._with_answer(t, f"本次生成的候选包括 {gen[0]}。最终推荐另一个分子 {OUTSIDE}；它从未出现在本次生成结果中。")
            self.assertFalse(r["pass"], t.id)
            self.assertLess(r["dimensions"]["answer"], 1.0, t.id)

    def test_recommending_both_a_generated_and_an_outside_molecule_fails(self):
        for t in self.design:
            a = self._generated(t)[0]
            self.assertFalse(self._with_answer(t, f"推荐 {a} 和 {OUTSIDE} 两个分子。")["pass"], t.id)

    def test_common_recommendation_formats_pass(self):
        for t in self.design:
            gen = self._generated(t)
            for answer in (f"**最终推荐：** `{gen[0]}`",
                           f"| 推荐候选 | {gen[0]} |",
                           f"本次生成的候选包括 {gen[0]}、{gen[1]}。综合比较后首选 {gen[1]}。",
                           f"硝胺类推荐 {gen[0]}；硝基芳烃类推荐 {gen[1]}。"):
                self.assertTrue(self._with_answer(t, answer)["pass"], (t.id, answer))


class TestTaskVersionOnResume(_TmpDir):
    """评审步骤：跑 A、B → 改 A 的判分条件 → 只续跑 B → 再续跑 A。"""

    def setUp(self):
        super().setUp()
        self.root = self.tmp / "tasks"
        for tid in ("name2smiles/tnt", "name2smiles/rdx"):
            shutil.copytree(TASKS_DIR / tid, self.root / tid)
        self.job = self.tmp / "job"
        run(OracleAgent(), [self.A(), self.B()], self.job)
        toml = self.root / "name2smiles/tnt/task.toml"
        toml.write_text(toml.read_text(encoding="utf-8").replace('name = "TNT"', 'name = "RDX"'), encoding="utf-8")

    def A(self):
        return load_task(self.root / "name2smiles/tnt", self.root)

    def B(self):
        return load_task(self.root / "name2smiles/rdx", self.root)

    def test_resuming_another_task_is_refused_while_any_task_is_stale(self):
        with self.assertRaises(JobError):
            run(OracleAgent(), [self.B()], self.job)
        with self.assertRaises(JobError):
            run(OracleAgent(), [self.A()], self.job)

    def test_stale_task_version_is_not_silently_refreshed(self):
        old = json.loads((self.job / "config.json").read_text(encoding="utf-8"))["tasks"]["name2smiles/tnt"]
        with self.assertRaises(JobError):
            run(OracleAgent(), [self.B()], self.job)
        now = json.loads((self.job / "config.json").read_text(encoding="utf-8"))["tasks"]["name2smiles/tnt"]
        self.assertEqual(old, now)
        self.assertNotEqual(now, self.A().content_hash())

    def test_rescore_applies_new_definition_then_resume_works(self):
        self.assertEqual(rescore_job(self.job, **QUIET)["overall"]["pass@1"], 0.5)
        self.assertEqual(run(OracleAgent(), [self.A(), self.B()], self.job)["overall"]["pass@1"], 0.5)

    def test_result_graded_under_another_version_is_rejected_at_aggregation(self):
        rescore_job(self.job, **QUIET)
        reward_path = trial_dir(self.job, "name2smiles/rdx", 0) / "reward.json"
        reward = json.loads(reward_path.read_text(encoding="utf-8"))
        reward["task_hash"] = "000000000000"
        reward_path.write_text(json.dumps(reward), encoding="utf-8")
        with self.assertRaises(JobError):
            run(OracleAgent(), [self.B()], self.job)


class TestNullIsNotEvidence(unittest.TestCase):
    """保留原调用和最终回答，只把工具结果换成 null 填充。"""

    def _tamper(self, task, mutate):
        traj = Trajectory.load(task.solution_path)
        step = traj.calls("generateAndPredict")[0]
        obj = json.loads(step.result)
        mutate(obj)
        step.result = json.dumps(obj)
        return verify(task, traj)

    @staticmethod
    def _props(task):
        return next(c for c in task.checks if c["type"] == "generation_output").get("required_properties") or []

    def test_reviewer_case_null_molecules_one_record_null_values(self):
        for t in load_tasks(["generate-predict"]):
            props = self._props(t)

            def mutate(obj):
                obj["generated_molecules"] = [None] * len(obj["generated_molecules"])
                obj["results"] = [{**{p: None for p in props}, "smiles": None}] if props else []
            r = self._tamper(t, mutate)
            self.assertFalse(r["pass"], t.id)
            self.assertLess(r["dimensions"]["l3_value"], 1.0, t.id)
            self.assertTrue(r["incomplete"], t.id)
            self.assertIn("gen_output", r["invalid_evidence"], t.id)

    def test_real_molecules_but_null_property_values_fail(self):
        for t in load_tasks(["generate-predict"]):
            props = self._props(t)
            if not props:
                continue

            def mutate(obj):
                for rec in obj["results"]:
                    rec.update({p: None for p in props})
            r = self._tamper(t, mutate)
            self.assertFalse(r["pass"], t.id)
            self.assertIn("gen_output", r["invalid_evidence"], t.id)

    def test_property_records_for_molecules_never_generated_do_not_count(self):
        for t in load_tasks(["generate-predict"]):
            if not self._props(t):
                continue

            def mutate(obj):
                for rec in obj["results"]:
                    rec["smiles"] = OUTSIDE
            self.assertFalse(self._tamper(t, mutate)["pass"], t.id)

    def test_design_tasks_with_null_candidates_fail_and_are_flagged(self):
        for t in load_tasks(["design-*"]):
            def mutate(obj):
                obj["generated_molecules"] = [None] * len(obj["generated_molecules"])
            r = self._tamper(t, mutate)
            self.assertFalse(r["pass"], t.id)
            self.assertTrue(r["incomplete"], t.id)

    def test_null_numeric_tool_result_is_flagged(self):
        t = load_tasks(["ob-kj/tnt"])[0]
        traj = Trajectory.load(t.solution_path)
        step = traj.calls("calculateKJ")[0]
        obj = json.loads(step.result)
        obj.update({k: None for k in ("DetoD", "DetoP", "DetoT", "DetoV", "DetoQ")})
        step.result = json.dumps(obj)
        r = verify(t, traj)
        self.assertFalse(r["pass"])
        self.assertIn("kj_values", r["invalid_evidence"])


if __name__ == "__main__":
    unittest.main()
