# src/skipper_ai/cli.py

import click
import os
import pandas as pd
from skipper_ai import ingest, train

@click.group()
def cli():
    """Skipper AI Assistant - Explain your sailing performance."""
    pass

@cli.command()
@click.argument('csv_path', type=click.Path(exists=True))
@click.option('--polar', default='data/polars/polaires_class40_2022.csv', help='Path to polar CSV')
@click.option('--run-id', default=None, help='Unique ID for this run (defaults to filename)')
def ingest_data(csv_path, polar, run_id):
    """Process raw telemetry and calculate performance metrics."""
    if run_id is None:
        run_id = os.path.splitext(os.path.basename(csv_path))[0]
    
    run_dir = os.path.join('data', 'runs', run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    click.echo(f"Processing {csv_path} with polar {polar}...")
    df = ingest.process_csv(csv_path, polar)
    
    output_path = os.path.join(run_dir, 'data.csv')
    df.to_csv(output_path, index=False)
    click.echo(f"✅ Ingestion complete. Run saved to {output_path}")

@cli.command()
@click.argument('run_id')
def analyze(run_id):
    """Run XGBoost and SHAP analysis on a run."""
    from skipper_ai.analyze import run_analysis
    
    run_dir = os.path.join('data', 'runs', run_id)
    data_path = os.path.join(run_dir, 'data.csv')
    
    if not os.path.exists(data_path):
        click.echo(f"❌ Error: Run data not found at {data_path}. Did you run 'ingest' first?")
        return

    click.echo(f"Analyzing run {run_id}...")
    analysis_results = run_analysis(data_path)
    
    import json
    results_path = os.path.join(run_dir, 'analysis.json')
    with open(results_path, 'w') as f:
        json.dump(analysis_results, f, indent=4)
    
    click.echo(f"✅ Analysis complete. Results saved to {results_path}")

@cli.command()
@click.argument('run_id')
def explain(run_id):
    """Generate a natural language explanation for a run."""
    from skipper_ai.explain import generate_explanation
    
    run_dir = os.path.join('data', 'runs', run_id)
    results_path = os.path.join(run_dir, 'analysis.json')
    
    if not os.path.exists(results_path):
        click.echo(f"❌ Error: Analysis results not found at {results_path}. Did you run 'analyze' first?")
        return

    click.echo(f"Generating explanation for run {run_id}...")
    explanation = generate_explanation(results_path)
    
    explanation_path = os.path.join(run_dir, 'explanation.txt')
    with open(explanation_path, 'w') as f:
        f.write(explanation)
    
    click.echo("\n--- Performance Explanation ---\n")
    click.echo(explanation)
    click.echo(f"\n✅ Explanation saved to {explanation_path}")

@cli.command()
@click.argument('run_id')
@click.argument('question')
def ask(run_id, question):
    """Ask a specific question about the run's performance."""
    from skipper_ai.explain import ask_question
    
    run_dir = os.path.join('data', 'runs', run_id)
    results_path = os.path.join(run_dir, 'analysis.json')
    
    if not os.path.exists(results_path):
        click.echo(f"❌ Error: Analysis results not found at {results_path}. Did you run 'analyze' first?")
        return

    click.echo(f"Answering question for run {run_id}: {question}")
    answer = ask_question(results_path, question)
    
    click.echo("\n--- Answer ---\n")
    click.echo(answer)

@cli.command("report")
@click.argument("run_id")
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    help="Open the report in the default browser after generation.",
)
@click.option(
    "--no-explain",
    is_flag=True,
    help="Skip loading or generating coach advice (numbers and charts only).",
)
def report_cmd(run_id, open_browser, no_explain):
    """Build an offline HTML report (charts + coach advice) for a run."""
    import webbrowser

    from skipper_ai.explain import generate_explanation
    from skipper_ai.report import build_report

    run_dir = os.path.join("data", "runs", run_id)
    data_path = os.path.join(run_dir, "data.csv")
    results_path = os.path.join(run_dir, "analysis.json")
    explanation_path = os.path.join(run_dir, "explanation.txt")

    if not os.path.exists(data_path):
        click.echo(
            f"❌ Error: Run data not found at {data_path}. Did you run 'ingest-data' first?"
        )
        return
    if not os.path.exists(results_path):
        click.echo(
            f"❌ Error: Analysis results not found at {results_path}. Did you run 'analyze' first?"
        )
        return

    if not no_explain and not os.path.exists(explanation_path):
        click.echo(f"Generating explanation for run {run_id}...")
        explanation = generate_explanation(results_path)
        with open(explanation_path, "w", encoding="utf-8") as f:
            f.write(explanation)
        click.echo(f"Explanation saved to {explanation_path}")

    click.echo(f"Building HTML report for run {run_id}...")
    out_path = build_report(run_dir, no_explain=no_explain)
    click.echo(f"✅ Report written to {out_path}")

    if open_browser:
        webbrowser.open(out_path.as_uri())


if __name__ == '__main__':
    cli()
