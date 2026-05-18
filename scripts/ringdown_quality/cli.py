from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import load_config, validate_config
from .reporting import run_pipeline


def parse_modes(raw: str) -> list[tuple[int, int]]:
    modes = []
    for token in raw.split():
        parts = token.split(",")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"Invalid mode token `{token}`; expected `ell,m`.")
        modes.append((int(parts[0]), int(parts[1])))
    return sorted(modes, key=lambda item: (item[0], item[1]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.ringdown_quality.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run the SXS ringdown-quality selector.")
    run.add_argument("--config", default="configs/default.yml", help="Path to the YAML configuration file.")
    run.add_argument("--output", default=None, help="Override output directory.")
    run.add_argument("--catalog-tag", default=None, help="Override the fixed SXS catalog tag.")
    run.add_argument("--modes", default=None, help='Override target modes, e.g. "2,2 2,1 3,3 4,4".')
    run.add_argument("--top-k", type=int, default=None, help="Override number of simulations to select.")
    run.add_argument("--max-simulations", type=int, default=None, help="Limit candidate simulations for a test run.")
    run.add_argument(
        "--allow-nonstandard-window",
        action="store_true",
        help="Developer/debug flag allowing a ringdown window other than [0,30]M.",
    )
    run.add_argument(
        "--allow-nonstandard-weights",
        action="store_true",
        help="Developer/debug flag allowing non-default metric weights.",
    )
    return parser


def _apply_overrides(config, args: argparse.Namespace):
    if args.output is not None:
        config.output.directory = str(Path(args.output))
    if args.catalog_tag is not None:
        config.catalog.tag = str(args.catalog_tag)
    if args.modes is not None:
        config.modes.target_modes = parse_modes(args.modes)
    if args.top_k is not None:
        config.selection.top_k = int(args.top_k)
        config.selection.top_fraction = None
        config.selection.min_common_score = None
    if args.max_simulations is not None:
        config.runtime.max_simulations = int(args.max_simulations)
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        config = load_config(
            args.config,
            allow_nonstandard_window=args.allow_nonstandard_window,
            allow_nonstandard_weights=args.allow_nonstandard_weights,
        )
        config = _apply_overrides(config, args)
        validate_config(
            config,
            allow_nonstandard_window=args.allow_nonstandard_window,
            allow_nonstandard_weights=args.allow_nonstandard_weights,
        )
        result = run_pipeline(config)
        selected = result["selected_simulations"]
        print(f"Wrote ringdown-quality outputs to {config.output.directory}")
        print(f"Selected {len(selected)} simulations")
        return 0
    parser.error(f"Unsupported command `{args.command}`.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
