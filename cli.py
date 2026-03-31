"""
Project DiMaggio — CLI Entry Point.

Usage:
  python cli.py picks --date 2025-06-15 --streak 23
  python cli.py predict --batter 660271 --date 2025-06-15
  python cli.py explain --batter 660271 --date 2025-06-15
  python cli.py pipeline --season 2024
  python cli.py train
"""

import logging
import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from config import LOG_LEVEL, BRONZE_DIR

logging.basicConfig(level=LOG_LEVEL)
console = Console()


@click.group()
def cli():
    """🎯 Project DiMaggio — MLB Beat the Streak Pick Engine"""
    pass


# ── picks ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--date",    required=True,  help="Game date YYYY-MM-DD")
@click.option("--streak",  default=0,      help="Current streak length", type=int)
@click.option("--dd",      default=1,      help="Double Down budget remaining", type=int)
@click.option("--savers",  default=1,      help="Streak Savers remaining", type=int)
@click.option("--min-p",   default=0.70,   help="Minimum P(Hit) threshold", type=float)
@click.option("--json",    "as_json", is_flag=True, help="Output raw JSON")
def picks(date, streak, dd, savers, min_p, as_json):
    """Show today's top picks with strategy applied."""
    from models.predict import rank_batters_for_date
    from strategy.milestone_logic import select_picks, get_threshold, get_phase
    from strategy.opener_detector import apply_opener_adjustment
    from strategy.shift_recalibration import apply_shift_recalibration
    from strategy.rl_agent import StreakDQNAgent, StreakState
    from config import MODEL_CHECKPOINT_DIR

    console.print(f"\n[bold cyan]🎯 Project DiMaggio — Picks for {date}[/bold cyan]")
    console.print(f"Streak: [bold]{streak}[/bold] | Phase: [bold]{get_phase(streak)}[/bold] | Threshold: P > [bold]{get_threshold(streak):.2f}[/bold]\n")

    with console.status("Ranking batters..."):
        candidates = rank_batters_for_date(date, min_prob=min_p)

    if not candidates:
        console.print("[yellow]⚠ No candidates found for this date. Is the pipeline run?[/yellow]")
        return

    candidates = apply_opener_adjustment(candidates, date)
    candidates = apply_shift_recalibration(candidates, date)
    candidates.sort(key=lambda c: c["p_hit"], reverse=True)

    selected = select_picks(candidates, streak, double_down_budget=dd)

    # RL recommendation
    rl_agent = StreakDQNAgent(epsilon=0.0)
    rl_path  = MODEL_CHECKPOINT_DIR / "rl_agent.pt"
    if rl_path.exists():
        rl_agent.load(rl_path)
    rl_state  = StreakState(
        streak_length=streak,
        p_hit=candidates[0]["p_hit"] if candidates else 0.75,
        double_downs_remaining=dd,
        streak_savers_remaining=savers,
    )
    rl_rec = rl_agent.recommend_action(rl_state)

    if as_json:
        output = {
            "date": date, "streak": streak,
            "picks": [p.to_dict() for p in selected],
            "rl_recommendation": rl_rec,
            "all_candidates": candidates[:10],
        }
        console.print_json(json.dumps(output))
        return

    # Rich table output
    table = Table(box=box.ROUNDED, title=f"Top Picks — {date}", show_lines=True)
    table.add_column("Rank",        style="bold yellow",  width=5)
    table.add_column("Batter ID",   style="cyan",         width=10)
    table.add_column("P(Hit)",      style="bold green",   width=8)
    table.add_column("Matchup",     style="white",        width=12)
    table.add_column("Stand",       style="white",        width=6)
    table.add_column("Double Down", style="bold red",     width=11)
    table.add_column("Notes",       style="dim",          width=40)

    for i, p in enumerate(selected):
        table.add_row(
            str(i + 1),
            str(p.batter_id),
            f"{p.p_hit:.3f}",
            f"{p.away_team} @ {p.home_team}",
            p.stand,
            "✅ YES" if p.double_down else "—",
            p.explanation[:60] if p.explanation else "",
        )

    console.print(table)

    # Show full candidate list
    if len(candidates) > len(selected):
        console.print(f"\n[dim]Also qualifying (not selected): "
                      f"{len(candidates) - len(selected)} batters[/dim]")

    console.print(Panel(
        f"[bold]RL Agent Recommendation:[/bold] {rl_rec['action_name'].upper()}\n"
        f"Q-values: {rl_rec['q_values']}",
        title="🤖 Strategic Advisor", border_style="blue"
    ))
    console.print()


# ── predict ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--batter", required=True, type=int,   help="Batter MLBAM ID")
@click.option("--date",   required=True,              help="Game date YYYY-MM-DD")
def predict(batter, date):
    """Get raw P(Hit) for a specific batter on a specific date."""
    from models.predict import predict_hit_prob

    with console.status(f"Predicting for batter {batter}..."):
        try:
            p = predict_hit_prob(batter, date)
            color = "green" if p >= 0.85 else "yellow" if p >= 0.75 else "red"
            console.print(
                f"Batter [cyan]{batter}[/cyan] on [cyan]{date}[/cyan]: "
                f"P(Hit) = [bold {color}]{p:.4f}[/bold {color}] ({p*100:.1f}%)"
            )
        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")


# ── explain ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--batter", required=True, type=int, help="Batter MLBAM ID")
@click.option("--date",   required=True,            help="Game date YYYY-MM-DD")
@click.option("--top-n",  default=5, type=int,      help="Number of top SHAP features")
def explain(batter, date, top_n):
    """Get a SHAP-based explanation for why a batter was picked."""
    from explainability.shap_explainer import shap_values_for_pick

    with console.status("Running SHAP analysis..."):
        result = shap_values_for_pick(batter, date, top_n=top_n)

    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        return

    console.print(Panel(
        f"[bold]Batter {batter} on {date}[/bold]\n"
        f"P(Hit) = {result['p_hit']:.4f}\n\n"
        f"[italic]{result['explanation_text']}[/italic]",
        title="🔍 SHAP Explanation", border_style="green"
    ))

    table = Table(box=box.SIMPLE, title="Top Contributing Features")
    table.add_column("Feature",     style="cyan")
    table.add_column("SHAP Value",  style="white")
    table.add_column("Direction",   style="green")
    table.add_column("Description", style="dim")

    for feat in result.get("top_features", []):
        direction = "↑ Boost" if feat["shap_value"] > 0 else "↓ Reduce"
        style     = "green" if feat["shap_value"] > 0 else "red"
        table.add_row(
            feat["feature"],
            f"{feat['shap_value']:+.4f}",
            f"[{style}]{direction}[/{style}]",
            feat["description"],
        )
    console.print(table)


# ── pipeline ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--season",     required=True, type=int, help="Season year (e.g. 2024)")
@click.option("--skip-ingest", is_flag=True,           help="Skip Statcast ingestion")
def pipeline(season, skip_ingest):
    """Run the full data pipeline: Bronze → Silver → Gold."""
    from bronze.ingest_statcast import ingest_statcast_season
    from bronze.ingest_weather import fetch_all_weather
    from silver.rolling_windows import run_silver_rolling
    from silver.pitcher_archetypes import run_pitcher_archetypes
    from silver.feature_engineering import run_silver_features
    from physics.park_factors import build_physics_df
    from gold.gold_table import run_gold_table
    import pandas as pd

    console.print(f"\n[bold cyan]🔧 Running Data Pipeline for {season}...[/bold cyan]\n")

    if not skip_ingest:
        with console.status("Ingesting Statcast data..."):
            paths = ingest_statcast_season(season)
        console.print(f"[green]✓ Statcast ingested: {len(paths)} chunks[/green]")
    else:
        console.print("[yellow]⏩ Skipping Statcast ingestion[/yellow]")

    with console.status("Building Silver rolling windows..."):
        run_silver_rolling(BRONZE_DIR)
    console.print("[green]✓ Silver rolling windows built[/green]")

    with console.status("Clustering pitcher archetypes..."):
        run_pitcher_archetypes()
    console.print("[green]✓ Pitcher archetypes (k=8) clustered[/green]")

    with console.status("Fetching live weather..."):
        try:
            weather = fetch_all_weather()
            weather_df = pd.DataFrame(list(weather.values()))
            weather_df.rename(columns={"team": "home_team"}, inplace=True)
        except Exception:
            console.print("[yellow]⚠ Weather fetch failed. Proceeding without.[/yellow]")
            weather_df = None

    with console.status("Building Silver features..."):
        run_silver_features(weather_df=weather_df)
    console.print("[green]✓ Silver features engineered[/green]")

    with console.status("Building Gold inference table..."):
        physics_df = build_physics_df(weather) if weather_df is not None else None
        run_gold_table(physics_df=physics_df)
    console.print("[green]✓ Gold table built[/green]")

    console.print("\n[bold green]🎉 Pipeline complete! Ready to train or run picks.[/bold green]")
    console.print("  Next step: [cyan]python cli.py train[/cyan]")


# ── train ──────────────────────────────────────────────────────────────────────

@cli.command()
def train():
    """Train the HTL model on the Gold table data."""
    from models.train import train_model

    console.print("\n[bold cyan]🧠 Training HTL Model...[/bold cyan]\n")
    try:
        ckpt_path = train_model()
        console.print(f"\n[bold green]✓ Model trained. Checkpoint: {ckpt_path}[/bold green]")
    except Exception as exc:
        console.print(f"\n[red]Training failed: {exc}[/red]")
        raise


if __name__ == "__main__":
    cli()
