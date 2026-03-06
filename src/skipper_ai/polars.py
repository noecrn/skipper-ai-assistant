# src/skipper_ai/polars.py

import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator

class PolarManager:
    def __init__(self, polar_path):
        """
        Loads Class40 polar data in the standard format: tws,sail_id,twa,boat_speed.
        """
        # Load the polar file
        df = pd.read_csv(polar_path)

        # Group by TWS and TWA to get the maximum boat speed for each combination
        df_optimal = df.groupby(['tws', 'twa'])['boat_speed'].max().reset_index()

        # Prepare the interpolation function
        self.points = df_optimal[['tws', 'twa']].values
        self.speeds = df_optimal['boat_speed'].values
        self.interp_func = LinearNDInterpolator(self.points, self.speeds)

    def get_expected_speed(self, tws, twa):
        """
        Calculates expected boat speed based on current wind conditions.
        """
        # Use absolute value of TWA for interpolation since polar data is symmetric
        twa_abs = abs(twa)
        
        # Interpolate the expected speed from the polar data
        expected = self.interp_func(tws, twa_abs)
        
        # If interpolation fails, return 0 or some default value
        if np.isnan(expected):
            return 0.0

        return float(expected)