"""
EMBench CLI（仓库根目录下执行）
==============================

    python -m embench list   [-t PATTERN ...]
    python -m embench check  [-t PATTERN ...]          # 题目自检（CI 用）
    python -m embench registry --write                 # 重建 registry.json
    python -m embench run    -a oracle                 # 参考答案（应 100%）
    python -m embench run    -a replay --replay-src <轨迹目录> --job my-agent
    python -m embench score  <job_dir>                 # 只重判，不重新答题
    python -m embench report <job_dir> [<job_dir> ...] # 单 job 摘要 / 多 job 对比
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from .agents import AGENT_NAMES, build_agent
from .core.checker import REGISTRY_FILE, check_registry, check_task, write_registry
from .core.report import compare_markdown, job_markdown
from .core.runner import JOBS_DIR, rescore_job, run_job
from .core.task import load_tasks


def cmd_list(args) -> int:
    tasks = load_tasks(args.tasks)
    for t in tasks:
        print(f"{t.id:45s} {t.meta.get('difficulty', ''):7s} {t.meta.get('description', '')}")
    fam = Counter(t.family for t in tasks)
    print(f"\n共 {len(tasks)} 题：" + "，".join(f"{f} {n}" for f, n in sorted(fam.items())))
    return 0


def cmd_check(args) -> int:
    tasks = load_tasks(args.tasks)
    bad = 0
    for t in tasks:
        rep = check_task(t.dir)
        if rep.ok:
            if args.verbose:
                print(f"  ok  {t.id}")
        else:
            bad += 1
            print(f"FAIL  {t.id}")
            for e in rep.errors:
                print(f"        - {e}")
    reg_errs = [] if args.tasks else check_registry()
    for e in reg_errs:
        print(f"REGISTRY  {e}")
    print(f"\n{len(tasks) - bad}/{len(tasks)} 题通过自检" + (f"，registry {len(reg_errs)} 处问题" if reg_errs else ""))
    return 1 if bad or reg_errs else 0


def cmd_registry(args) -> int:
    if args.write:
        reg = write_registry()
        print(f"已写入 {REGISTRY_FILE}（{reg['n_tasks']} 题，{len(reg['families'])} 个 family）")
        return 0
    errs = check_registry()
    print("\n".join(errs) or "registry.json 是最新的")
    return 1 if errs else 0


def cmd_run(args) -> int:
    tasks = load_tasks(args.tasks)
    if not tasks:
        print("没有匹配的任务")
        return 1
    agent = build_agent(args)
    job = args.job or f"{datetime.datetime.now():%Y%m%d_%H%M%S}_{args.agent}"
    job_dir = Path(args.jobs_dir) / job
    result = asyncio.run(run_job(agent, tasks, job_dir, n_trials=args.n_trials,
                                 n_concurrent=args.n_concurrent, force=args.force))
    print("\n" + job_markdown(job_dir))
    print(f"结果目录: {job_dir}")
    return 0 if result["overall"]["n_trials"] else 1


def cmd_score(args) -> int:
    job_dir = Path(args.job_dir)
    rescore_job(job_dir)
    print(job_markdown(job_dir))
    return 0


def cmd_report(args) -> int:
    dirs = [Path(d) for d in args.job_dirs]
    text = job_markdown(dirs[0]) if len(dirs) == 1 else compare_markdown(dirs)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"已写入 {args.out}")
    else:
        print(text)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="embench", description="EMBench 化学智能体工具调用评测基准")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_tasks(sp):
        sp.add_argument("-t", "--tasks", nargs="*", default=None,
                        help="任务 id glob 或 family 名（缺省全部），如 'ob-kj/*' name2smiles")

    sp = sub.add_parser("list", help="列出任务")
    add_tasks(sp)
    sp.set_defaults(fn=cmd_list)

    sp = sub.add_parser("check", help="题目自检：结构 + oracle 必过 + nop/截断必挂 + registry")
    add_tasks(sp)
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(fn=cmd_check)

    sp = sub.add_parser("registry", help="校验 / 重建 registry.json")
    sp.add_argument("--write", action="store_true")
    sp.set_defaults(fn=cmd_registry)

    sp = sub.add_parser("run", help="跑一个 job")
    add_tasks(sp)
    sp.add_argument("-a", "--agent", choices=AGENT_NAMES, required=True)
    sp.add_argument("--replay-src", nargs="*", default=None, help="replay 的轨迹目录（结构：<family>/<instance>.json）")
    sp.add_argument("-k", "--n-trials", type=int, default=1, help="每题重复次数（>1 时报告 pass@k / pass^k）")
    sp.add_argument("-n", "--n-concurrent", type=int, default=1)
    sp.add_argument("--job", default=None, help="job 名（缺省按时间戳）；同名再跑 = 续跑")
    sp.add_argument("--jobs-dir", default=str(JOBS_DIR))
    sp.add_argument("--force", action="store_true", help="忽略已有结果全部重跑")
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("score", help="用当前任务定义重判一个已有 job（不调 LLM）")
    sp.add_argument("job_dir")
    sp.set_defaults(fn=cmd_score)

    sp = sub.add_parser("report", help="单 job 摘要或多 job 对比")
    sp.add_argument("job_dirs", nargs="+")
    sp.add_argument("--out", default=None)
    sp.set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
