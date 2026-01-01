import pandas as pd

df = pd.read_csv('trajectories_generated_exact_region.csv')
print(f"Total Unique: {df['VehicleID'].nunique()}")
print(f"Total Rows: {len(df)}")
