import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

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

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- Hyperparameter Grid ----
param_grid = {
    'n_estimators': [200, 300, 400],  # Number of trees
    'learning_rate': [0.01, 0.05, 0.1],  # Step size
    'max_depth': [4, 6, 8],  # Tree depth
    'subsample': [0.7, 0.8, 1.0],  # % of data used per tree
    'colsample_bytree': [0.7, 0.8, 1.0],  # % of features per tree
}

# XGBoost Model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Grid Search for Best Hyperparameters
grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=1)

grid_search.fit(X_train_scaled, y_train)

# Best Model
best_xgb = grid_search.best_estimator_

# Make predictions
best_preds = best_xgb.predict(X_test_scaled)

# Evaluate accuracy
best_accuracy = accuracy_score(y_test, best_preds)

print(f"Best XGBoost Accuracy After Tuning: {best_accuracy:.4f}")
print(f"Best Parameters: {grid_search.best_params_}")
