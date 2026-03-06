# src/skipper_ai/ingest.py

import pandas as pd
import numpy as np
from skipper_ai.polars import PolarManager

def process_csv(csv_path, polar_model_path):
    """
    Processes a Virtual Regatta CSV file to calculate performance ratios.
    """
    # Initialize PolarManager
    pm = PolarManager(polar_model_path)

    # Load the CSV data
    df = pd.read_csv(csv_path)

    # Calculate Expected Performance using the real data from the CSV
    df['expected_speed'] = df.apply(
        lambda x: pm.get_expected_speed(x['tws'], x['twa']), axis=1
    )

    # Calculate Performance Ratio
    df['performance_ratio'] = df['boat_speed'] / df['expected_speed'].replace(0, np.nan)
    df['performance_ratio'] = df['performance_ratio'].fillna(0.0)

    return df

if __name__ == "__main__":
    # Ensure these paths match your folder structure
    polar_csv = "data/polars/polaires_class40_2022.csv"
    input_csv = "data/raw/virtual_regatta.csv"
    output_csv = "data/processed/run_sardinia.csv"

    try:
        df_processed = process_csv(input_csv, polar_csv)
        df_processed.to_csv(output_csv, index=False)
        print(f"✅ Ingestion complete. Processed {len(df_processed)} rows.")
        print(df_processed[['timestamp', 'boat_speed', 'expected_speed', 'performance_ratio']].head())
    except FileNotFoundError as e:
        print(f"❌ Error: {e}. Check that your CSV files are in the data/ folder.")