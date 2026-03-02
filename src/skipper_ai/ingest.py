# src/skipper_ai/ingest.py

import pandas as pd
import numpy as np
import gpxpy
from skipper_ai.polars import PolarManager

def process_gpx(gpx_path, polat_model_path):
	"""
	Processes a GPX file to extract sailing performance data and calculates performance ratios based on a polar model.
	"""
	# Initialize PolarManager with the provided polar model path
	pm = PolarManager(polat_model_path)

	# Read the GPX file and extract relevant data
	with open(gpx_path, 'r') as gpx_file:
		gpx = gpxpy.parse(gpx_file)

	# Extract data points from the GPX file
	data = []
	for track in gpx.tracks:
		for segment in track.segments:
			# Iterate through the points in the segment to extract timestamp, latitude, longitude, and calculate boat speed
			for i in range(1, len(segment.points)):
				p1 = segment.points[i-1]
				p2 = segment.points[i]
		
				# Calculate distance in meters between p1 and p2
				distance = p2.distance_2d(p1)
		
				# Calculate time difference in seconds between p1 and p2
				time_delta = (p2.time - p1.time).total_seconds()
		
				if time_delta > 0:
					# Convert speed from m/s to knots (1 m/s = 1.94384 knots)
					speed_kts = (distance / time_delta) * 1.94384
				else:
					continue  # Skip if time difference is zero to avoid division by zero
		
				data.append({
                    'timestamp': p2.time,
                    'lat': p2.latitude,
                    'lon': p2.longitude,
                    'boat_speed': speed_kts, # Vitesse calculée
                    'tws': 15.0, # Placeholder POC
                    'twa': 45.0, # Placeholder POC
                    'heel': 20.0, # Placeholder POC
                    'sail_id': 'J1' # Placeholder POC
                })

	df = pd.DataFrame(data)

	# Calculate expected speed using the polar model
	df['expected_speed'] = df.apply(
		lambda x: pm.get_expected_speed(x['tws'], x['twa']), axis=1
	)

	# Calculate performance ratio (actual speed / expected speed)
	df['performance_ratio'] = df['boat_speed'] / df['expected_speed'].replace(0, np.nan)
	df.fillna(0, inplace=True)

	return df

if __name__ == "__main__":
	# Process the GPX file and save the resulting DataFrame to a CSV file
	final_df = process_gpx("data/raw/sardinia_trip.gpx", "data/polars/polaires_class40_2022.csv")
	final_df.to_csv("data/processed/run_sardinia.csv", index=False)
	print("✅ Ingestion complete. Dataset saved to data/processed/")