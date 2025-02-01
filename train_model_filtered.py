import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
file_path = "charity_data.csv"
df = pd.read_csv(file_path)

# Drop non-beneficial columns & low-impact features
df = df.drop(columns=['EIN', 'NAME', 'SPECIAL_CONSIDERATIONS', 'STATUS'])

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features and target
X = df.drop(columns=['IS_SUCCESSFUL']).values
y = df['IS_SUCCESSFUL'].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- REFINED NEURAL NETWORK ----
nn_filtered = Sequential()

# Input Layer
nn_filtered.add(Dense(units=128, activation=LeakyReLU(alpha=0.01), input_shape=(X_train.shape[1],)))

# Hidden Layers
nn_filtered.add(Dense(units=64, activation=LeakyReLU(alpha=0.01)))
nn_filtered.add(Dropout(0.1))

nn_filtered.add(Dense(units=32, activation=LeakyReLU(alpha=0.01)))
nn_filtered.add(Dropout(0.1))

# Output Layer
nn_filtered.add(Dense(units=1, activation='sigmoid'))

# Compile the model
nn_filtered.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

# Train the model
history_filtered = nn_filtered.fit(X_train_scaled, y_train,
                                   epochs=100,
                                   batch_size=32,
                                   validation_split=0.2,
                                   verbose=2)

# Evaluate final model performance
loss_filtered, accuracy_filtered = nn_filtered.evaluate(X_test_scaled, y_test, verbose=2)

# Save final trained model
nn_filtered.save("AlphabetSoupCharity_Filtered.keras")

print(f"Final Accuracy After Feature Selection: {accuracy_filtered:.4f}")
print("Model saved as AlphabetSoupCharity_Filtered.keras")
