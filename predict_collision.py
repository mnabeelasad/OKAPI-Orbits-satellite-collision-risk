# predict_collision.py

import pandas as pd
import pickle

# 💾 Dataset load karo
df = pd.read_pickle('dataset.pkl')

# 🔍 Unique satellite pairs dikhayein
pairs = df[['sat1_name', 'sat2_name']].drop_duplicates()
print("Available Satellite Pairs:")
print(pairs)

# 🔽 User input lein
sat1 = input("\nEnter Satellite 1 name: ").strip()
sat2 = input("Enter Satellite 2 name: ").strip()

# ✅ Select matching row
row = df[(df['sat1_name'] == sat1) & (df['sat2_name'] == sat2)]

if row.empty:
    print("❌ Pair not found in dataset. Check names.")
    exit()

sample = row.iloc[0]

# 🎯 Input features extract karo
input_data = sample[['rel_distance', 'rel_velocity', 'inclo_diff', 'ecco_diff', 'raan_diff']].astype(float).to_frame().T

# 💾 Model load karo
with open('satellite_collision_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 🔮 Prediction
prediction = model.predict(input_data)[0]
result = "⚠️ RISKY (Collision Possible)" if prediction == 1 else "✅ SAFE (No Collision Risk)"

# 📋 Detailed Output
print(f"\nPrediction for {sat1} vs {sat2}: {result}")
print("\n📊 Feature Values Used:")
print(f"📍 Relative Distance:  {sample['rel_distance']:.2f} km")
print(f"📍 Relative Velocity:  {sample['rel_velocity']:.2f} km/s")
print(f"📍 Inclination Diff:   {sample['inclo_diff']:.2f} deg")
print(f"📍 Eccentricity Diff:  {sample['ecco_diff']:.6f}")
print(f"📍 RAAN Diff:          {sample['raan_diff']:.2f} deg")

# 🧠 Explanation (Basic Logic)
if prediction == 0:
    print("\n🔎 Reason:")
    if sample['rel_distance'] >= 100:
        print("✔️ Distance is large enough (>= 100 km), so considered SAFE.")
    else:
        print("✔️ Other orbital parameters suggest no strong collision risk.")
else:
    print("\n🔎 Reason:")
    print("❗ Close distance and/or strong orbital alignment indicates collision risk.")
