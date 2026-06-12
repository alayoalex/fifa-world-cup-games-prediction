"""
CLI for custom hypothetical matches.

Examples:
    uv run python src/etl/add_fixture.py add --home Spain --away Brazil
    uv run python src/etl/add_fixture.py add --home Argentina --away France --date 2026-07-15
    uv run python src/etl/add_fixture.py list
    uv run python src/etl/add_fixture.py remove --id abc123def456
    uv run python src/etl/add_fixture.py predict
    uv run python src/etl/add_fixture.py add --home Spain --away Brazil --predict
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from etl import build_match_table  # noqa: E402
from etl import custom_fixtures  # noqa: E402
from features import build_features  # noqa: E402
from models.predict_fixtures import predict_custom  # noqa: E402


def _parse_date(value: str) -> pd.Timestamp:
    return pd.to_datetime(value).normalize()


def _cmd_add(args: argparse.Namespace) -> None:
    custom_fixtures.add(
        args.home,
        args.away,
        match_date=_parse_date(args.date) if args.date else None,
        neutral=not args.home_advantage,
        city=args.city or "",
        country=args.country or "",
        force=args.force,
        skip_if_exists=args.predict and not args.force,
    )
    if args.predict:
        _rebuild_and_predict(args.output)


def _cmd_list(_: argparse.Namespace) -> None:
    fixtures = custom_fixtures.list_fixtures()
    if fixtures.empty:
        print("[custom_fixtures] no custom fixtures stored")
        return
    show = fixtures[
        ["fixture_id", "date", "home_team", "away_team", "neutral", "city", "country"]
    ].copy()
    show["date"] = show["date"].dt.date
    print(show.to_string(index=False))


def _cmd_remove(args: argparse.Namespace) -> None:
    custom_fixtures.remove(args.id)
    if args.rebuild:
        _rebuild_features()


def _cmd_clear(args: argparse.Namespace) -> None:
    custom_fixtures.clear()
    if args.rebuild:
        _rebuild_features()


def _rebuild_features() -> None:
    print("\n=== rebuild match table + features ===")
    build_match_table.build()
    build_features.build()


def _rebuild_and_predict(output: Path | None) -> None:
    _rebuild_features()
    predict_custom(output=output)


def _cmd_predict(args: argparse.Namespace) -> None:
    if custom_fixtures.list_fixtures().empty:
        raise SystemExit("No custom fixtures. Add one with: add_fixture.py add --home X --away Y")
    _rebuild_and_predict(args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage custom hypothetical football fixtures")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="add a hypothetical fixture")
    p_add.add_argument("--home", required=True, help="home team (canonical or alias)")
    p_add.add_argument("--away", required=True, help="away team (canonical or alias)")
    p_add.add_argument("--date", help="match date YYYY-MM-DD (default: day after latest result)")
    p_add.add_argument(
        "--home-advantage", action="store_true",
        help="disable neutral venue (apply home advantage in Elo)",
    )
    p_add.add_argument("--city", default="", help="optional host city")
    p_add.add_argument("--country", default="", help="optional host country")
    p_add.add_argument("--predict", action="store_true", help="rebuild features and predict")
    p_add.add_argument(
        "--force", action="store_true",
        help="replace the fixture if the same home/away/date already exists",
    )
    p_add.add_argument(
        "--output", type=Path, default=None,
        help="prediction CSV path (default: data/processed/custom_predictions.csv)",
    )
    p_add.set_defaults(func=_cmd_add)

    p_list = sub.add_parser("list", help="list stored custom fixtures")
    p_list.set_defaults(func=_cmd_list)

    p_remove = sub.add_parser("remove", help="remove a custom fixture by id")
    p_remove.add_argument("--id", required=True, help="fixture_id from list command")
    p_remove.add_argument(
        "--rebuild", action="store_true",
        help="rebuild match table and features after removal",
    )
    p_remove.set_defaults(func=_cmd_remove)

    p_clear = sub.add_parser("clear", help="remove all custom fixtures")
    p_clear.add_argument("--rebuild", action="store_true", help="rebuild after clearing")
    p_clear.set_defaults(func=_cmd_clear)

    p_predict = sub.add_parser("predict", help="rebuild features and score custom fixtures")
    p_predict.add_argument("--output", type=Path, default=None)
    p_predict.set_defaults(func=_cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
