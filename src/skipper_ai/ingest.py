# src/skipper_ai/ingest.py

import pandas as pd
import numpy as np
import gpxpy
from skipper_ai.polars import PolarManager

def process_gpx(gpx_path, polat_model_path):
	# Initialize PolarManager with the provided polar model path
	pm = PolarManager(polat_model_path)

	# Read the GPX file and extract relevant data
	with open(gpx_path, 'r') as gpx_file:
		gpx = gpxpy.parse(gpx_file)

	# Extract data points from the GPX file
	data = []
	for track in gpx.tracks:
		for segment in track.segments:
			for point in segment.points:
				data.append({
					'timestamp': point.time,
					'lat': point.latitude,
					'lon': point.longitude,
					'boat_speed': point.speed if point.speed else 0.0
				})

	# Convert the extracted data into a DataFrame
	df = pd.DataFrame(data)

	# For simplicity, we will use fixed values for TWS and TWA. In a real application, these would be derived from the data or provided as input.
	df['tws'] = 15.0
	df['twa'] = 45.0
	df['sail_mode'] = 'J1'

	# Calculate expected speed using the polar model
	df['expected_speed'] = df.apply(lambda row: pm.get_expected_speed(row['tws'], row['twa']), axis=1)

	# Calculate expected speed using the polar model
	df['performance_ratio'] = df['boat_speed'] / df['expected_speed'].replace(0, np.nan)
	df['performance_ratio'] = df['performance_ratio'].fillna(0.0)

	return df

if __name__ == "__main__":
	# Process the GPX file and save the resulting DataFrame to a CSV file
	final_df = process_gpx("data/raw/sardinia_trip.gpx", "data/polars/polaires_class40_2022.csv")
	final_df.to_csv("data/processed/run_sardinia.csv", index=False)
	print("✅ Ingestion complete. Dataset saved to data/processed/")