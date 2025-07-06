import streamlit as st
import pandas as pd
import pickle

# 💾 Load trained model
with open('satellite_collision_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 📄 Load dataset
df = pd.read_pickle('dataset.pkl')

# 🔍 Get all unique satellite pairs
pairs = df[['sat1_name', 'sat2_name']].drop_duplicates().reset_index(drop=True)

st.title("🚀 Satellite Collision Risk Predictor")

st.markdown("Select two satellites below to check for potential collision risk.")

# 🧭 Satellite pair selection dropdowns
sat1_options = pairs['sat1_name'].unique().tolist()
selected_sat1 = st.selectbox("Select Satellite 1", sat1_options)

# Filter Satellite 2 options based on Satellite 1
filtered_pairs = pairs[pairs['sat1_name'] == selected_sat1]
sat2_options = filtered_pairs['sat2_name'].unique().tolist()
selected_sat2 = st.selectbox("Select Satellite 2", sat2_options)

if st.button("Predict Collision Risk"):
    row = df[(df['sat1_name'] == selected_sat1) & (df['sat2_name'] == selected_sat2)]

    if row.empty:
        st.error("❌ Pair not found. Try different satellites.")
    else:
        sample = row.iloc[0]
        input_data = sample[['rel_distance', 'rel_velocity', 'inclo_diff', 'ecco_diff', 'raan_diff']] \
                        .astype(float).to_frame().T
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error(f"⚠️ RISKY: Collision Risk between {selected_sat1} and {selected_sat2}")
        else:
            st.success(f"✅ SAFE: No Collision Risk between {selected_sat1} and {selected_sat2}")

        # 📊 Show detailed feature values
        st.markdown("### 📊 Feature Values Used:")
        st.write({
            "Relative Distance (km)": round(sample['rel_distance'], 2),
            "Relative Velocity (km/s)": round(sample['rel_velocity'], 2),
            "Inclination Difference (deg)": round(sample['inclo_diff'], 2),
            "Eccentricity Difference": round(sample['ecco_diff'], 6),
            "RAAN Difference (deg)": round(sample['raan_diff'], 2)
        })

        # 💡 Add reasoning explanation
        st.markdown("### 🔎 Reason for Prediction:")
        if prediction == 0:
            if sample['rel_distance'] >= 100:
                st.info("✔️ Distance is large enough (≥ 100 km), so considered **SAFE**.")
            else:
                st.info("✔️ Other orbital parameters indicate no strong collision risk.")
        else:
            st.warning("❗ Close distance and/or similar orbits indicate **collision risk**.")
