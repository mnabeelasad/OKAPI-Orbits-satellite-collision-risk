
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # ✅ Random Forest import
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import xgboost as xgb

# 💾 Dataset load kar rahe hain jo data_generator.py ne banaya tha
df = pd.read_pickle('dataset.pkl')

# 🧠 Features (X) aur Target (y) define kar rahe hain
X = df[['rel_distance', 'rel_velocity', 'inclo_diff', 'ecco_diff', 'raan_diff']]
y = df['risk']

# 📦 Data ko training aur testing mein split karte hain
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # ✅ Random Forest model initialize kar rahe hain
# model = RandomForestClassifier(
#     n_estimators=100,       # 100 trees banenge
#     max_depth=6,            # Har tree ki max depth 6 hogi (optional tuning)
#     random_state=42         # Reproducibility ke liye
# )

# XGBoost model initialize kar rahe hain
model = xgb.XGBClassifier(
    eval_metric='logloss',   # Binary classification ke liye
    use_label_encoder=False, # Deprecation warning avoid karne ke liye
    n_estimators=100,        # 100 boosting rounds
    max_depth=6,             # Har tree ki max depth 6
    random_state=42          # Reproducibility ke liye
)


# 🚀 Model train karte hain
model.fit(X_train, y_train)

# 🔍 Test data pe prediction le rahe hain
y_pred = model.predict(X_test)

# 📊 Model ki accuracy check karte hain
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.3f}")
print("📋 Classification Report:")
print(classification_report(y_test, y_pred))

# 💾 Trained model ko save karte hain taake prediction script use kar sake
with open('satellite_collision_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained & saved as 'satellite_collision_model.pkl'")
