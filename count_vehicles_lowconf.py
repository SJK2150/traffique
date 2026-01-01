import pandas as pd

df = pd.read_csv('trajectories_lowconf.csv')
unique_vehicles = df['VehicleID'].nunique()
total_rows = len(df)

# Get count by class if available
class_counts = df['Class'].value_counts() if 'Class' in df.columns else "N/A"

print(f"Total Unique Vehicles: {unique_vehicles}")
print(f"Total Detections: {total_rows}")
print("\nBreakdown by Class ID (if available):")
print(class_counts)
