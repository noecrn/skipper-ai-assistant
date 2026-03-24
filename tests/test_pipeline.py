# tests/test_pipeline.py

import os
import pandas as pd
from skipper_ai import ingest, analyze

def test_ingestion():
    polar_path = 'data/polars/polaires_class40_2022.csv'
    raw_path = 'data/raw/synthetic_run.csv'
    
    if not os.path.exists(raw_path):
        print(f"Skipping test_ingestion: {raw_path} not found.")
        return
        
    df = ingest.process_csv(raw_path, polar_path)
    assert 'expected_speed' in df.columns
    assert 'performance_ratio' in df.columns
    assert 'sail_id_numeric' in df.columns
    print("test_ingestion passed!")

def test_analysis():
    run_id = 'synthetic_run'
    data_path = f'data/runs/{run_id}/data.csv'
    
    if not os.path.exists(data_path):
        print(f"Skipping test_analysis: {data_path} not found.")
        return
        
    summary = analyze.run_analysis(data_path)
    assert 'feature_importance' in summary
    assert 'avg_performance' in summary
    print("test_analysis passed!")

if __name__ == "__main__":
    test_ingestion()
    test_analysis()
