import pandas as pd

df = pd.read_csv('trajectories_lowconf_highres.csv')
unique = df['VehicleID'].nunique()
rows = len(df)
print(f"Total Unique: {unique}")
print(f"Total Rows: {rows}")
