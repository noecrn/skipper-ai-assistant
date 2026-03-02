import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator

class PolarManager:
    def __init__(self, polar_path):
        # Load the polar file
        df = pd.read_csv(polar_path, sep=';')

        # Extract unique TWS and TWA values
        tws = sorted(df['tws'].unique())
        twa = sorted(df['twa'].unique())

        # Create a matrix of speeds with TWA as rows and TWS as columns
        matrix = df.pivot(index='twa', columns='tws', values='speed').values

        # Create an interpolation function
        self.interp_func = RegularGridInterpolator(
            (tws, twa), 
            matrix, 
            method='linear'
            bounds_error=False,
            fill_value=None
        )

    def get_expected_speed(self, tws, twa):
        # Normalize TWA to the range [0, 180]
        twa_normalized = abs(((twa + 180) % 360) - 180)

        # Interpolate the speed for the given TWS and normalized TWA
        point = np.array([twa_normalized, tws])
        return float(self.interp_func(point))