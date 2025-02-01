import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
file_path = "charity_data.csv"
df = pd.read_csv(file_path)

# Drop non-beneficial columns
df = df.drop(columns=['EIN', 'NAME'])

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features and target
X = df.drop(columns=['IS_SUCCESSFUL'])
y = df['IS_SUCCESSFUL']

# Apply SMOTE to balance dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- Final XGBoost Model ----
xgb_final = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    colsample_bytree=0.8,
    learning_rate=0.02,  # Increased from 0.01
    max_depth=10,  # Increased from 8
    n_estimators=500,  # Increased from 400
    subsample=0.8,
    random_state=42
)

xgb_final.fit(X_train_scaled, y_train)

# Make predictions
xgb_preds_final = xgb_final.predict(X_test_scaled)

# Evaluate accuracy
xgb_final_accuracy = accuracy_score(y_test, xgb_preds_final)

print(f"🔥 Final XGBoost Accuracy After Fine-Tuning: {xgb_final_accuracy:.4f}")
