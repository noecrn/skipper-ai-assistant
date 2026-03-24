# src/skipper_ai/generate_synthetic.py

import pandas as pd
import numpy as np
import os
from skipper_ai.polars import PolarManager

def generate_synthetic_data(polar_path, output_path, n_rows=100):
    """
    Generates synthetic sailing data with injected performance losses.
    """
    pm = PolarManager(polar_path)
    
    # Random TWS and TWA
    tws = np.random.uniform(5, 25, n_rows)
    twa = np.random.uniform(30, 180, n_rows)
    
    # Base heel (proportional to TWS)
    heel = tws * 1.5 + np.random.normal(0, 2, n_rows)
    
    # Sail selection (simplified: Jib if TWA < 110, else Spi)
    sail_id = np.where(twa < 110, 'Jib', 'Spi')
    
    data = []
    for i in range(n_rows):
        expected = pm.get_expected_speed(tws[i], twa[i])
        
        # Injected losses
        loss = 1.0
        
        # Loss 1: Over-heeling (heel > 25)
        if heel[i] > 25:
            loss *= 0.85 + np.random.normal(0, 0.02)
            
        # Loss 2: Wrong sail (Spi when upwind, Jib when downwind)
        if (twa[i] < 90 and sail_id[i] == 'Spi') or (twa[i] > 140 and sail_id[i] == 'Jib'):
            loss *= 0.70 + np.random.normal(0, 0.05)
            
        # Some random noise
        loss *= np.random.normal(1.0, 0.02)
        
        boat_speed = expected * loss
        
        data.append({
            'timestamp': 1700000000 + i * 60,
            'lat': 45.0 + i * 0.01,
            'lon': -1.0 + i * 0.01,
            'boat_speed': boat_speed,
            'tws': tws[i],
            'twa': twa[i],
            'heel': heel[i],
            'sail_id': sail_id[i]
        })
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Synthetic data generated at {output_path}")

if __name__ == "__main__":
    generate_synthetic_data('data/polars/polaires_class40_2022.csv', 'data/raw/synthetic_run.csv')
