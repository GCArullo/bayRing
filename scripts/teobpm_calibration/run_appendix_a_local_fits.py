#!/usr/bin/env python3
"""Run the reusable Appendix A local-fit workflow.

This wrapper intentionally delegates to ``scripts.teobpm_calibration.teobpm_calibration`` instead
of storing generated per-job shell commands.  It is meant to be run from any
checkout with explicit campaign paths.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _default_worker_count() -> int:
    return max(1, os.cpu_count() or 1)


def _add_common_environment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pythonpath", default=None, help="Optional PYTHONPATH to prepend while running bayRing.")
    parser.add_argument("--numba-cache-dir", default=None, help="Optional NUMBA_CACHE_DIR for numba/SXS imports.")
    parser.add_argument("--mplconfigdir", default=None, help="Optional MPLCONFIGDIR for headless plotting.")
    parser.add_argument("--sxsconfigdir", default=None, help="Optional SXSCONFIGDIR.")
    parser.add_argument("--sxscachedir", default=None, help="Optional SXSCACHEDIR.")


def _env_from_args(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.pythonpath:
        env["PYTHONPATH"] = args.pythonpath + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else args.pythonpath
    for option_name, env_name in (
        ("numba_cache_dir", "NUMBA_CACHE_DIR"),
        ("mplconfigdir", "MPLCONFIGDIR"),
        ("sxsconfigdir", "SXSCONFIGDIR"),
        ("sxscachedir", "SXSCACHEDIR"),
    ):
        value = getattr(args, option_name)
        if value:
            env[env_name] = value
    return env


def _run(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, env=env)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True, help="Appendix A campaign directory to create or reuse.")
    parser.add_argument("--nc-ringdown-dir", default=None, help="Path to the nc_ringdown checkout.")
    parser.add_argument("--workers", type=int, default=_default_worker_count(), help="Number of parallel bayRing local-fit workers.")
    parser.add_argument("--timeout", type=float, default=None, help="Per-job timeout in seconds.")
    parser.add_argument("--bayring-executable", default="bayRing", help="bayRing executable used by run-local-fits.")
    parser.add_argument("--skip-prepare", action="store_true", help="Do not run appendix-a-prepare.")
    parser.add_argument("--skip-run", action="store_true", help="Do not run local fits.")
    parser.add_argument("--skip-collect", action="store_true", help="Do not collect local-fit outputs.")
    parser.add_argument("--skip-global-fit", action="store_true", help="Do not construct Appendix A global fits.")
    parser.add_argument("--skip-plot", action="store_true", help="Do not generate Appendix A diagnostic plots.")
    parser.add_argument("--skip-compare", action="store_true", help="Do not generate nonspinning comparison plots.")
    parser.add_argument("--force", action="store_true", help="Re-run existing local-fit outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    _add_common_environment_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    campaign_dir = Path(args.campaign_dir)
    env = _env_from_args(args)
    module = [sys.executable, "-m", "scripts.teobpm_calibration.teobpm_calibration"]

    if not args.skip_prepare:
        command = module + ["appendix-a-prepare", "--output-dir", str(campaign_dir), "--bayring-executable", args.bayring_executable]
        if args.nc_ringdown_dir:
            command += ["--nc-ringdown-dir", args.nc_ringdown_dir]
        _run(command, env, args.dry_run)

    if not args.skip_run:
        command = module + [
            "run-local-fits",
            "--campaign-dir",
            str(campaign_dir),
            "--workers",
            str(args.workers),
            "--bayring-executable",
            args.bayring_executable,
        ]
        if args.force:
            command.append("--force")
        if args.timeout is not None:
            command += ["--timeout", str(args.timeout)]
        _run(command, env, args.dry_run)

    if not args.skip_collect:
        _run(module + ["appendix-a-collect", "--campaign-dir", str(campaign_dir)], env, args.dry_run)

    if not args.skip_global_fit:
        _run(module + ["appendix-a-global-fit", "--campaign-dir", str(campaign_dir)], env, args.dry_run)

    if not args.skip_plot:
        _run(module + ["appendix-a-plot", "--campaign-dir", str(campaign_dir)], env, args.dry_run)

    if not args.skip_compare:
        command = module + ["appendix-a-compare-nonspinning", "--campaign-dir", str(campaign_dir)]
        if args.nc_ringdown_dir:
            command += ["--nc-ringdown-dir", args.nc_ringdown_dir]
        _run(command, env, args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
