"""Markdown 报告：单 job 摘要 + 多 job 横向对比（排行榜）。"""
from __future__ import annotations

import json
from pathlib import Path

DIM_ORDER = ["l1_tool", "l2_args", "l3_value", "coverage", "consistency", "answer", "efficiency"]


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _num(x, fmt="{:.1f}") -> str:
    return "—" if x is None else fmt.format(x)


def _load(job_dir: Path) -> tuple[dict, dict]:
    cfg = json.loads((job_dir / "config.json").read_text(encoding="utf-8"))
    res = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    return cfg, res


def _label(cfg: dict) -> str:
    a = cfg["agent"]
    parts = [a.get("name")] + [str(a[k]) for k in ("model", "mode") if a.get(k)]
    return " / ".join(parts)


def job_markdown(job_dir: Path) -> str:
    cfg, res = _load(job_dir)
    o = res["overall"]
    env = cfg["environment"]
    k = o["k"]
    lines = [
        f"# EMBench job `{cfg['job']}`",
        "",
        f"- agent: **{_label(cfg)}**",
        f"- git: `{env['git_commit']}`{' (dirty)' if env['git_dirty'] else ''} · python {env['python']} · rdkit {env['rdkit']}",
        f"- 任务 {o['n_tasks']} 道 × {k} 次 = {o['n_trials']} trial；错误 {o['error_trials']}；缺少证据 {o['incomplete_trials']}"
        + (f"；⚠ 重复轨迹 {o['duplicate_trials']}" if o.get("duplicate_trials") else "")
        + (f"；⚠ 无 agent 身份记录 {o['unverified_agent_trials']}" if o.get("unverified_agent_trials") else ""),
        "",
        f"**pass@1 = {_pct(o['pass@1'])}**（Wilson 95% CI {_pct(o['wilson_95'][0])} – {_pct(o['wilson_95'][1])}，按 {o['n_tasks']} 道题计）"
        + (f" · pass@{k} = {_pct(o.get(f'pass@{k}'))} · pass^{k} = {_pct(o.get(f'pass^{k}'))}" if k > 1 else ""),
        "",
        "| family | 题数 | pass@1 | " + " | ".join(DIM_ORDER) + " | 调用数 | 冗余 | 耗时(s) |",
        "|---|---:|---:|" + "---:|" * len(DIM_ORDER) + "---:|---:|---:|",
    ]
    for fam, s in list(res["families"].items()) + [("**overall**", o)]:
        dims = " | ".join(_pct(s["dimensions"].get(d)) for d in DIM_ORDER)
        c = s["cost"]
        lines.append(f"| {fam} | {s['n_tasks']} | {_pct(s['pass@1'])} | {dims} | "
                     f"{_num(c['tool_calls'])} | {_num(c['redundancy'])} | {_num(c['wall_time_s'])} |")

    fails = {tid: v for tid, v in res["per_task"].items() if not v["pass"]}
    if fails:
        lines += ["", f"## 未通过（{len(fails)}）", ""]
        for tid, v in fails.items():
            lines.append(f"- `{tid}` — " + ("; ".join(v["failed_checks"]) or "required 检查被跳过"))
    return "\n".join(lines) + "\n"


def compare_markdown(job_dirs: list[Path]) -> str:
    loaded = [(d, *_load(d)) for d in job_dirs]
    families = sorted({f for _, _, r in loaded for f in r["families"]})
    head = "| family | " + " | ".join(f"{_label(c)}<br>`{d.name}`" for d, c, _ in loaded) + " |"
    lines = ["# EMBench 对比（pass@1）", "", head, "|---|" + "---:|" * len(loaded)]
    for fam in families:
        cells = [_pct(r["families"].get(fam, {}).get("pass@1")) for _, _, r in loaded]
        lines.append(f"| {fam} | " + " | ".join(cells) + " |")
    lines.append("| **overall** | " + " | ".join(f"**{_pct(r['overall']['pass@1'])}**" for _, _, r in loaded) + " |")
    lines += ["", "## 维度均值", "", "| 维度 | " + " | ".join(d.name for d, _, _ in loaded) + " |",
              "|---|" + "---:|" * len(loaded)]
    for dim in DIM_ORDER:
        lines.append(f"| {dim} | " + " | ".join(_pct(r["overall"]["dimensions"].get(dim)) for _, _, r in loaded) + " |")
    lines.append("| 平均调用数 | " + " | ".join(_num(r["overall"]["cost"]["tool_calls"]) for _, _, r in loaded) + " |")
    lines.append("| 平均耗时(s) | " + " | ".join(_num(r["overall"]["cost"]["wall_time_s"]) for _, _, r in loaded) + " |")
    lines.append("| 平均输出 token | " + " | ".join(_num(r["overall"]["cost"]["output_tokens"], "{:.0f}") for _, _, r in loaded) + " |")
    return "\n".join(lines) + "\n"
