# data_generator.py

import numpy as np
import pandas as pd
from skyfield.api import load
from datetime import timedelta

# Load TLE data
satellite_data = load.tle_file("space-data.tle")[:50]
ts = load.timescale()

print(f"Loaded satellites: {[sat.name for sat in satellite_data]}")

data = []

# Satellite pairs ka loop
for i, sat1 in enumerate(satellite_data):
    for j, sat2 in enumerate(satellite_data[i+1:], start=i+1):
        try:
            epoch = sat1.epoch
            t_base = ts.utc(epoch.utc.year, epoch.utc.month, epoch.utc.day, epoch.utc.hour, epoch.utc.minute, epoch.utc.second)

            inclo1 = sat1.model.inclo * 180.0 / np.pi
            inclo2 = sat2.model.inclo * 180.0 / np.pi
            ecco1 = sat1.model.ecco
            ecco2 = sat2.model.ecco
            raan1 = sat1.model.nodeo * 180.0 / np.pi
            raan2 = sat2.model.nodeo * 180.0 / np.pi

            for minutes in np.linspace(0, 24*60, 48):
                t = ts.utc(t_base.utc_datetime() + timedelta(minutes=minutes))
                pos1 = sat1.at(t).position.km
                pos2 = sat2.at(t).position.km
                vel1 = sat1.at(t).velocity.km_per_s
                vel2 = sat2.at(t).velocity.km_per_s

                if np.any(np.isnan(pos1)) or np.any(np.isnan(pos2)) or np.any(np.isnan(vel1)) or np.any(np.isnan(vel2)):
                    continue

                rel_distance = np.linalg.norm(pos1 - pos2)
                rel_velocity = np.linalg.norm(vel1 - vel2)

                noise = np.random.normal(0, 0.1, 5)
                rel_distance_noisy = max(0, rel_distance + noise[0] * rel_distance * 0.1)
                rel_velocity_noisy = max(0, rel_velocity + noise[1] * rel_velocity * 0.1)
                inclo_diff = abs(inclo1 - inclo2) + noise[2]
                ecco_diff = abs(ecco1 - ecco2) + noise[3] * 0.0001
                raan_diff = abs(raan1 - raan2) + noise[4]

                label = 1 if rel_distance_noisy < 3000 else 0

                features = {
                    'sat1_name': sat1.name,
                    'sat2_name': sat2.name,
                    'rel_distance': rel_distance_noisy,
                    'rel_velocity': rel_velocity_noisy,
                    'inclo_diff': inclo_diff,
                    'ecco_diff': ecco_diff,
                    'raan_diff': raan_diff,
                    'risk': label
                }
                data.append(features)
        except Exception as e:
            print(f"Error: {e}")

# DataFrame bana ke save karo
df = pd.DataFrame(data)

# Agar sirf ek class ho to risky cases add karo
if len(df['risk'].unique()) < 2:
    risky_rows = df.sample(frac=0.1, random_state=42).copy()
    risky_rows['risk'] = 1
    risky_rows['rel_distance'] = np.random.uniform(5, 50, size=len(risky_rows))
    df = pd.concat([df, risky_rows], ignore_index=True)

df.to_pickle('dataset.pkl')
print("✅ Dataset generated and saved to dataset.pkl")
