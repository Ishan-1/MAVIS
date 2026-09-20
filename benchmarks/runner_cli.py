"""
benchmarks/runner_cli.py
Command-line interface for MAVIS-Bench.
Supports task listing, single persona evaluation, dry-run testing, and score reporting.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from benchmarks.harness.runner import BenchmarkRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mavis-bench",
        description="MAVIS-Bench: Multi-persona benchmark for proactive agent evaluation.",
    )
    parser.add_argument(
        "--persona",
        type=str,
        default=None,
        help="Run tasks for a specific persona (e.g. daily_driver, oss_maintainer, office_worker) or 'all'.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Run a specific task ID directly (requires --persona).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered personas and tasks without executing.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=30,
        help="Maximum turn budget for simulated user dialogue (default: 30).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run verification of task specs and workspaces without invoking live LLMs.",
    )
    return parser.parse_args()


def list_personas_and_tasks(bench_dir: Path) -> None:
    data_dir = bench_dir / "data"
    if not data_dir.exists():
        print(f"No benchmark data directory found at {data_dir}")
        return

    print("\n==================== MAVIS-Bench Registry ====================")
    personas = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    for p in personas:
        profile_file = data_dir / p / "profile.yaml"
        desc = ""
        if profile_file.exists():
            import yaml
            try:
                with open(profile_file, "r") as f:
                    p_data = yaml.safe_load(f)
                desc = f" - {p_data.get('name', '')}"
            except Exception:
                pass
        print(f"\n[Persona] {p}{desc}")

        tasks_dir = data_dir / p / "tasks"
        if tasks_dir.exists():
            tasks = sorted([t.name for t in tasks_dir.iterdir() if t.is_dir()])
            for t in tasks:
                t_yaml = tasks_dir / t / "task.yaml"
                title = ""
                if t_yaml.exists():
                    import yaml
                    try:
                        with open(t_yaml, "r") as f:
                            t_data = yaml.safe_load(f)
                        title = f" ({t_data.get('title', '')})"
                    except Exception:
                        pass
                print(f"  * {t}{title}")
    print("\n==============================================================")


async def main_async() -> int:
    args = parse_args()
    bench_dir = Path("benchmarks")

    if args.list or (not args.persona and not args.task):
        list_personas_and_tasks(bench_dir)
        return 0

    mavis_runner_fn = None
    llm_client = None
    if not args.dry_run:
        from benchmarks.harness.mavis_adapter import MavisBenchmarkAdapter
        adapter = MavisBenchmarkAdapter()
        mavis_runner_fn = adapter.execute_turn
        llm_client = adapter.llm_client

    runner = BenchmarkRunner(
        bench_root=bench_dir,
        max_turns=args.max_turns,
        llm_client=llm_client,
        mavis_runner_fn=mavis_runner_fn,
    )

    if args.task:
        if not args.persona:
            print("Error: --persona must be specified when using --task")
            return 1
        print(f"\n[MAVIS-Bench] Running single task: {args.persona} -> {args.task}")
        task_spec = runner.load_task(args.persona, args.task)
        _, profile = runner.load_episode(args.persona)
        result = await runner.run_task(task_spec, profile)

        print("\n--- Task Result ---")
        print(f"Status:        {result.status}")
        print(f"Turns Taken:   {result.turns_taken}")
        print(f"PROC Score:    {result.proc_score:.2f}")
        print(f"COMP Score:    {result.comp_score:.2f}")
        print(f"HEAL Score:    {result.heal_score:.2f}")
        print(f"SAFE Score:    {result.safe_score:.2f}")
        print(f"Verify Output: {result.verification_output}")
        return 0 if result.verification_passed else 1

    target_personas = (
        [d.name for d in (bench_dir / "data").iterdir() if d.is_dir()]
        if args.persona == "all"
        else [args.persona]
    )

    print(f"\n[MAVIS-Bench] Executing episode(s) for: {', '.join(target_personas)}")
    for persona in target_personas:
        print(f"\n>>> Running Episode: {persona}")
        ep_result = await runner.run_episode(persona)
        print(f"Finished {persona}: Composite Score = {ep_result.overall_score:.3f} (COMP: {ep_result.overall_comp:.2f}, PROC: {ep_result.overall_proc:.2f})")

    return 0


def main() -> None:
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
