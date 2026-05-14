import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("FRAUD DETECTION MODEL TRAINING")
print("=" * 60)

# Create models folder
os.makedirs('models', exist_ok=True)

# Step 1: Load data
print("\n[1/6] Loading data...")
df = pd.read_csv('data/transactions.csv')
print(f"   ✅ Loaded {len(df):,} transactions")
print(f"   📊 Fraud rate: {df['is_fraud'].mean()*100:.4f}%")

# Step 2: Feature engineering
print("\n[2/6] Engineering features...")
df['amount_log'] = np.log1p(df['amount'])
df['combined_risk'] = (df['device_risk_score'] * 0.6 + df['ip_risk_score'] * 0.4) / 100
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['is_high_risk'] = (df['combined_risk'] > 0.7).astype(int)
print(f"   ✅ Created 5 new features")

# Step 3: Encode categorical variables
print("\n[3/6] Encoding categorical variables...")
le_transaction = LabelEncoder()
le_merchant = LabelEncoder()

df['transaction_type_encoded'] = le_transaction.fit_transform(df['transaction_type'])
df['merchant_category_encoded'] = le_merchant.fit_transform(df['merchant_category'])
print(f"   ✅ Transaction types: {list(le_transaction.classes_)}")
print(f"   ✅ Merchant categories: {list(le_merchant.classes_)}")

# Step 4: Prepare features
print("\n[4/6] Preparing feature matrix...")
feature_columns = [
    'amount', 'amount_log', 'hour', 'day_of_week', 
    'device_risk_score', 'ip_risk_score', 'is_foreign',
    'combined_risk', 'is_high_risk', 'hour_sin', 'hour_cos',
    'transaction_type_encoded', 'merchant_category_encoded'
]

X = df[feature_columns]
y = df['is_fraud']
print(f"   ✅ Features: {len(feature_columns)} columns")
print(f"   ✅ X shape: {X.shape}")

# Step 5: Split and scale data
print("\n[5/6] Splitting and scaling data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"   ✅ Train size: {len(X_train):,}")
print(f"   ✅ Test size: {len(X_test):,}")

# Step 6: Handle imbalance with SMOTE and train
print("\n[6/6] Training model with SMOTE...")
print(f"   ⚖️  Original fraud rate: {y_train.mean()*100:.2f}%")

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"   ⚖️  After SMOTE fraud rate: {y_train_resampled.mean()*100:.2f}%")

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_resampled, y_train_resampled)

# Save all artifacts
print("\n" + "=" * 60)
print("SAVING MODEL ARTIFACTS")
print("=" * 60)

joblib.dump(model, 'models/fraud_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le_transaction, 'models/label_encoder_transaction.pkl')
joblib.dump(le_merchant, 'models/label_encoder_merchant.pkl')
joblib.dump(feature_columns, 'models/feature_columns.pkl')

print("✅ fraud_model.pkl - Trained Random Forest model")
print("✅ scaler.pkl - StandardScaler for normalization")
print("✅ label_encoder_transaction.pkl - Transaction type encoder")
print("✅ label_encoder_merchant.pkl - Merchant category encoder")
print("✅ feature_columns.pkl - Feature names list")

# Evaluate model
print("\n" + "=" * 60)
print("MODEL EVALUATION ON TEST DATA")
print("=" * 60)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n📊 Confusion Matrix:")
print(f"   True Negatives (correct legit):  {tn:,}")
print(f"   False Positives (false alarms):  {fp:,}")
print(f"   False Negatives (missed fraud):  {fn:,}")
print(f"   True Positives (caught fraud):   {tp:,}")

# Key Metrics
fraud_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"\n📈 Performance Metrics:")
print(f"   Accuracy:           {accuracy*100:.2f}%")
print(f"   Fraud Recall:       {fraud_recall*100:.2f}%  (Target: >80%)")
print(f"   False Positive Rate:{false_positive_rate*100:.2f}%  (Target: <5%)")
print(f"   Precision:          {precision*100:.2f}%")
print(f"   ROC-AUC:            {roc_auc:.4f}")

print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE! Model saved to 'models/' folder")
print("=" * 60)