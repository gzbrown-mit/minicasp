#!/usr/bin/env python
"""
Run MCTS retrosynthesis search using the AiZynthFinder-derived minicasp modules.

Usage:
    python scripts/run_mcts.py --config configs/mcts_config.yml --target "CC(=O)Oc1ccccc1C(=O)O"
    python scripts/run_mcts.py --config configs/mcts_config.yml --targets_file targets.txt
"""
from __future__ import annotations

import argparse
import json
import os
import time

from minicasp.stock.config import Configuration
from minicasp.search.search import MctsSearchTree
from minicasp.analysis.tree_analysis import TreeAnalysis
from minicasp.utils.logging import setup_logger


def run_search(config: Configuration, target_smiles: str) -> dict:
    """Run MCTS search for a single target and return results dict."""
    tree = MctsSearchTree(config, root_smiles=target_smiles)
    time_limit = config.search.time_limit
    iteration_limit = config.search.iteration_limit

    t0 = time.time()
    solved = False
    for i in range(iteration_limit):
        elapsed = time.time() - t0
        if elapsed > time_limit:
            break
        solved = tree.one_iteration()
        if solved and config.search.return_first:
            break
    elapsed = time.time() - t0

    analysis = TreeAnalysis(tree)
    try:
        best = analysis.best()
        best_score = analysis.scorers[0](best)
    except ValueError:
        best = None
        best_score = 0.0

    return {
        "target": target_smiles,
        "solved": solved,
        "score": float(best_score),
        "iterations": tree.profiling["iterations"],
        "time_s": round(elapsed, 2),
        "expansion_calls": tree.profiling["expansion_calls"],
    }


def main():
    parser = argparse.ArgumentParser(description="Run MCTS retrosynthesis search")
    parser.add_argument(
        "--config", required=True, help="Path to YAML configuration file"
    )
    parser.add_argument("--target", default=None, help="Single target SMILES string")
    parser.add_argument(
        "--targets_file",
        default=None,
        help="Path to file with one SMILES per line",
    )
    parser.add_argument(
        "--output", default=None, help="Path to write JSON results (default: stdout)"
    )
    args = parser.parse_args()

    if not args.target and not args.targets_file:
        parser.error("Provide either --target or --targets_file")

    setup_logger(console_level=20)  # INFO

    print(f"Loading configuration from {args.config}")
    config = Configuration.from_file(args.config)
    print(f"  Search: algorithm={config.search.algorithm}, "
          f"max_transforms={config.search.max_transforms}, "
          f"iteration_limit={config.search.iteration_limit}, "
          f"time_limit={config.search.time_limit}s")
    print(f"  Expansion policies: {list(config.expansion_policy.items)}")
    print(f"  Stock: {list(config.stock.items)}")

    targets = []
    if args.target:
        targets.append(args.target)
    if args.targets_file:
        with open(args.targets_file) as f:
            for line in f:
                smi = line.strip()
                if smi and not smi.startswith("#"):
                    targets.append(smi)

    print(f"\nRunning MCTS on {len(targets)} target(s)...\n")
    results = []
    n_solved = 0
    for i, smi in enumerate(targets):
        print(f"[{i + 1}/{len(targets)}] {smi[:80]}...")
        result = run_search(config, smi)
        results.append(result)
        status = "SOLVED" if result["solved"] else "FAILED"
        print(f"  {status} | score={result['score']:.3f} | "
              f"iters={result['iterations']} | time={result['time_s']}s")
        if result["solved"]:
            n_solved += 1

    summary = {
        "n_targets": len(targets),
        "n_solved": n_solved,
        "success_rate": n_solved / max(1, len(targets)),
        "results": results,
    }

    print(f"\n{'=' * 50}")
    print(f"Summary: {n_solved}/{len(targets)} solved "
          f"({summary['success_rate']:.1%})")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results written to {args.output}")
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
