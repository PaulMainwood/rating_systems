"""
Command-line interface for rating systems.

Usage:
    python -m rating_systems fit <data.parquet> [options]
    python -m rating_systems top <data.parquet> [options]
    python -m rating_systems predict <data.parquet> <player1> <player2> [options]
    python -m rating_systems matchup <data.parquet> <player1> <player2> [options]
    python -m rating_systems backtest <data.parquet> [options]
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import polars as pl


def load_player_names(path: Optional[str]) -> Optional[dict]:
    """Load player names from a parquet file."""
    if path is None:
        return None
    df = pl.read_parquet(path)
    if "rating_idx" in df.columns and "name" in df.columns:
        return dict(zip(df["rating_idx"].to_list(), df["name"].to_list()))
    if "player_id" in df.columns and "name" in df.columns:
        return dict(zip(df["player_id"].to_list(), df["name"].to_list()))
    return None


def cmd_fit(args):
    """Fit a rating system and optionally save results."""
    from ..data import GameDataset
    from ..systems import Elo, Glicko, Glicko2

    dataset = GameDataset.from_parquet(args.data)
    player_names = load_player_names(args.players)

    print(f"Loaded {dataset}")

    # Select system
    systems = {
        "elo": lambda: Elo(k_factor=args.k_factor),
        "glicko": lambda: Glicko(),
        "glicko2": lambda: Glicko2(),
    }

    if args.system not in systems:
        print(f"Unknown system: {args.system}")
        print(f"Available: {', '.join(systems.keys())}")
        return 1

    system = systems[args.system]()
    print(f"Fitting {system.__class__.__name__}...")

    system.fit(dataset, player_names=player_names)
    fitted = system.get_fitted_ratings()

    print(f"\n{fitted}")

    if args.output:
        fitted.save(args.output)
        print(f"\nSaved to {args.output}")

    if args.top:
        print(f"\nTop {args.top} players:")
        print(fitted.top(args.top))

    return 0


def cmd_top(args):
    """Show top N players."""
    from ..data import GameDataset
    from ..systems import Elo, Glicko, Glicko2

    dataset = GameDataset.from_parquet(args.data)
    player_names = load_player_names(args.players)

    systems = {
        "elo": lambda: Elo(k_factor=args.k_factor),
        "glicko": lambda: Glicko(),
        "glicko2": lambda: Glicko2(),
    }

    system = systems.get(args.system, lambda: Elo())()
    system.fit(dataset, player_names=player_names)
    fitted = system.get_fitted_ratings()

    print(f"Top {args.n} players ({args.system.upper()}):\n")
    print(fitted.top(args.n))

    return 0


def cmd_predict(args):
    """Predict outcome between two players."""
    from ..data import GameDataset
    from ..systems import Elo, Glicko, Glicko2

    dataset = GameDataset.from_parquet(args.data)
    player_names = load_player_names(args.players)

    systems = {
        "elo": lambda: Elo(k_factor=args.k_factor),
        "glicko": lambda: Glicko(),
        "glicko2": lambda: Glicko2(),
    }

    system = systems.get(args.system, lambda: Elo())()
    system.fit(dataset, player_names=player_names)
    fitted = system.get_fitted_ratings()

    p1, p2 = args.player1, args.player2

    # Try to resolve names to IDs
    if player_names:
        name_to_id = {v: k for k, v in player_names.items()}
        if isinstance(p1, str) and p1 in name_to_id:
            p1 = name_to_id[p1]
        if isinstance(p2, str) and p2 in name_to_id:
            p2 = name_to_id[p2]

    p1, p2 = int(p1), int(p2)
    prob = fitted.predict(p1, p2)

    name1 = fitted.get_name(p1)
    name2 = fitted.get_name(p2)

    print(f"\nMatchup Prediction ({args.system.upper()}):")
    print(f"  {name1} vs {name2}")
    print(f"  P({name1} wins) = {prob:.1%}")
    print(f"  P({name2} wins) = {1-prob:.1%}")

    return 0


def cmd_matchup(args):
    """Detailed matchup analysis."""
    from ..data import GameDataset
    from ..systems import Elo, Glicko, Glicko2

    dataset = GameDataset.from_parquet(args.data)
    player_names = load_player_names(args.players)

    systems = {
        "elo": lambda: Elo(k_factor=args.k_factor),
        "glicko": lambda: Glicko(),
        "glicko2": lambda: Glicko2(),
    }

    system = systems.get(args.system, lambda: Elo())()
    system.fit(dataset, player_names=player_names)
    fitted = system.get_fitted_ratings()

    p1, p2 = int(args.player1), int(args.player2)

    print(f"\nMatchup Analysis ({args.system.upper()}):\n")
    print(fitted.matchup(p1, p2))

    return 0


def cmd_backtest(args):
    """Run backtest on a dataset."""
    from ..data import GameDataset
    from ..systems import Elo, Glicko, Glicko2
    from ..evaluation import Backtester, compare_systems

    dataset = GameDataset.from_parquet(args.data)

    print(f"Dataset: {dataset}")
    print(f"Train fraction: {args.train_fraction}")

    # Determine split day
    days = dataset.days
    split_idx = int(len(days) * args.train_fraction)
    train_end_day = days[split_idx - 1] if split_idx > 0 else days[0]

    if args.system == "all":
        systems = [
            Elo(k_factor=16),
            Elo(k_factor=32),
            Glicko(),
            Glicko2(),
        ]
        results = compare_systems(systems, dataset, train_end_day=train_end_day)
        print("\nResults:")
        print(results)
    else:
        systems_map = {
            "elo": Elo(k_factor=args.k_factor),
            "glicko": Glicko(),
            "glicko2": Glicko2(),
        }
        system = systems_map.get(args.system, Elo())
        backtester = Backtester(system, dataset)
        result = backtester.run(train_end_day=train_end_day)
        print(result.summary())

    return 0


def cmd_matrix(args):
    """Generate head-to-head probability matrix."""
    from ..data import GameDataset
    from ..systems import Elo, Glicko

    dataset = GameDataset.from_parquet(args.data)
    player_names = load_player_names(args.players)

    system = Elo(k_factor=args.k_factor) if args.system == "elo" else Glicko()
    system.fit(dataset, player_names=player_names)
    fitted = system.get_fitted_ratings()

    matrix = fitted.head_to_head_matrix(top_n=args.n)

    print(f"\nHead-to-Head Win Probabilities (Top {args.n}):\n")
    print(matrix)

    if args.output:
        matrix.write_csv(args.output)
        print(f"\nSaved to {args.output}")

    return 0


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Rating Systems CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common arguments
    def add_common_args(p):
        p.add_argument("data", help="Path to games parquet file")
        p.add_argument("--players", "-p", help="Path to player names parquet file")
        p.add_argument("--system", "-s", default="elo",
                       choices=["elo", "glicko", "glicko2"],
                       help="Rating system to use")
        p.add_argument("--k-factor", "-k", type=float, default=32.0,
                       help="K-factor for Elo (default: 32)")

    # fit command
    fit_parser = subparsers.add_parser("fit", help="Fit a rating system")
    add_common_args(fit_parser)
    fit_parser.add_argument("--output", "-o", help="Save fitted ratings to file")
    fit_parser.add_argument("--top", "-t", type=int, default=10,
                            help="Show top N players (default: 10)")

    # top command
    top_parser = subparsers.add_parser("top", help="Show top N players")
    add_common_args(top_parser)
    top_parser.add_argument("-n", type=int, default=10, help="Number of players")

    # predict command
    predict_parser = subparsers.add_parser("predict", help="Predict matchup outcome")
    add_common_args(predict_parser)
    predict_parser.add_argument("player1", help="Player 1 ID or name")
    predict_parser.add_argument("player2", help="Player 2 ID or name")

    # matchup command
    matchup_parser = subparsers.add_parser("matchup", help="Detailed matchup analysis")
    add_common_args(matchup_parser)
    matchup_parser.add_argument("player1", type=int, help="Player 1 ID")
    matchup_parser.add_argument("player2", type=int, help="Player 2 ID")

    # backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("data", help="Path to games parquet file")
    backtest_parser.add_argument("--system", "-s", default="all",
                                 choices=["elo", "glicko", "glicko2", "all"],
                                 help="System to backtest (default: all)")
    backtest_parser.add_argument("--train-fraction", "-f", type=float, default=0.7,
                                 help="Fraction of data for training (default: 0.7)")
    backtest_parser.add_argument("--k-factor", "-k", type=float, default=32.0,
                                 help="K-factor for Elo")

    # matrix command
    matrix_parser = subparsers.add_parser("matrix", help="Head-to-head probability matrix")
    add_common_args(matrix_parser)
    matrix_parser.add_argument("-n", type=int, default=10,
                               help="Number of top players to include")
    matrix_parser.add_argument("--output", "-o", help="Save matrix to CSV")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    commands = {
        "fit": cmd_fit,
        "top": cmd_top,
        "predict": cmd_predict,
        "matchup": cmd_matchup,
        "backtest": cmd_backtest,
        "matrix": cmd_matrix,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
