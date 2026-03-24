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

if __name__ == '__main__':
    cli()
