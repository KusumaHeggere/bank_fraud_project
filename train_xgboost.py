import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("XGBOOST FRAUD DETECTION MODEL TRAINING")
print("=" * 60)

# Create models folder
os.makedirs('models', exist_ok=True)

# Check if data exists
if not os.path.exists('data/transactions.csv'):
    print("❌ Data file not found! Please run: python generate_synthetic_data.py first")
    exit(1)

# Load data
print("\n[1/5] Loading data...")
df = pd.read_csv('data/transactions.csv')
print(f"   ✅ Loaded {len(df):,} transactions")
print(f"   📊 Fraud rate: {df['is_fraud'].mean()*100:.4f}%")

# Feature engineering
print("\n[2/5] Engineering features...")
df['amount_log'] = np.log1p(df['amount'])
df['combined_risk'] = (df['device_risk_score'] * 0.6 + df['ip_risk_score'] * 0.4) / 100
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Encode categorical variables
le_type = LabelEncoder()
le_merchant = LabelEncoder()
df['type_encoded'] = le_type.fit_transform(df['transaction_type'])
df['merchant_encoded'] = le_merchant.fit_transform(df['merchant_category'])

# Prepare features
features = ['amount', 'amount_log', 'hour', 'day_of_week', 'device_risk_score', 
            'ip_risk_score', 'is_foreign', 'combined_risk', 'hour_sin', 'hour_cos',
            'type_encoded', 'merchant_encoded']

X = df[features]
y = df['is_fraud']

print(f"   ✅ Features: {len(features)} columns")

# Split data
print("\n[3/5] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle imbalance with SMOTE
print("\n[4/5] Handling class imbalance with SMOTE...")
print(f"   Original fraud rate: {y_train.mean()*100:.2f}%")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"   After SMOTE fraud rate: {y_train_resampled.mean()*100:.2f}%")

# Train XGBoost model
print("\n[5/5] Training XGBoost model...")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train_resampled, y_train_resampled)

# Evaluate
print("\n" + "=" * 60)
print("MODEL EVALUATION")
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

# Metrics
fraud_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"\n📈 Performance Metrics:")
print(f"   Accuracy: {accuracy*100:.2f}%")
print(f"   Fraud Recall: {fraud_recall*100:.2f}%")
print(f"   False Positive Rate: {false_positive_rate*100:.2f}%")
print(f"   ROC-AUC: {roc_auc:.4f}")

# Print classification report
print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

# Find optimal threshold
print("\n🎯 Finding optimal threshold for business costs...")

# Business costs: Missing fraud (FN) is 10x more costly than false alarm (FP)
COST_FN = 10  # Cost of missing a fraud
COST_FP = 1   # Cost of false alarm

best_threshold = 0.5
best_cost = float('inf')

for threshold in np.arange(0.1, 0.95, 0.05):
    y_pred_thresh = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
    
    # Calculate total business cost
    total_cost = (fn * COST_FN) + (fp * COST_FP)
    
    if total_cost < best_cost:
        best_cost = total_cost
        best_threshold = threshold

print(f"   ✅ Optimal threshold: {best_threshold:.2f}")
print(f"   📊 Estimated business cost at this threshold: {best_cost}")

# Also calculate F1 score optimization for reference
best_f1 = 0
best_f1_threshold = 0.5

for threshold in np.arange(0.1, 0.95, 0.05):
    y_pred_thresh = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    if f1 > best_f1:
        best_f1 = f1
        best_f1_threshold = threshold

print(f"   📊 F1-optimal threshold: {best_f1_threshold:.2f} (F1 Score: {best_f1:.3f})")

# Save model and artifacts
print("\n" + "=" * 60)
print("SAVING MODEL ARTIFACTS")
print("=" * 60)

joblib.dump(model, 'models/xgboost_fraud_model.pkl')
joblib.dump(scaler, 'models/scaler_xgb.pkl')
joblib.dump(le_type, 'models/le_type_xgb.pkl')
joblib.dump(le_merchant, 'models/le_merchant_xgb.pkl')
joblib.dump(features, 'models/features_xgb.pkl')
joblib.dump(best_threshold, 'models/best_threshold.pkl')

print("✅ xgboost_fraud_model.pkl - XGBoost model")
print("✅ scaler_xgb.pkl - StandardScaler for normalization")
print("✅ le_type_xgb.pkl - Transaction type encoder")
print("✅ le_merchant_xgb.pkl - Merchant category encoder")
print("✅ features_xgb.pkl - Feature names list")
print("✅ best_threshold.pkl - Optimal threshold for predictions")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE! Model saved to 'models/' folder")
print("=" * 60)

# Test with sample transactions
print("\n🔍 Testing with sample transactions...")

test_transactions = [
    {"name": "Normal Transaction", "amount": 45.00, "hour": 14, "day_of_week": 3, 
     "device_risk_score": 15, "ip_risk_score": 20, "is_foreign": 0, 
     "transaction_type": "pos", "merchant_category": "grocery"},
    
    {"name": "Suspicious Transaction", "amount": 2500.00, "hour": 3, "day_of_week": 6,
     "device_risk_score": 95, "ip_risk_score": 90, "is_foreign": 1,
     "transaction_type": "wire", "merchant_category": "travel"}
]

for test in test_transactions:
    # Create DataFrame
    test_df = pd.DataFrame([test])
    
    # Feature engineering
    test_df['amount_log'] = np.log1p(test_df['amount'])
    test_df['combined_risk'] = (test_df['device_risk_score'] * 0.6 + test_df['ip_risk_score'] * 0.4) / 100
    test_df['hour_sin'] = np.sin(2 * np.pi * test_df['hour'] / 24)
    test_df['hour_cos'] = np.cos(2 * np.pi * test_df['hour'] / 24)
    
    # Encode
    test_df['type_encoded'] = le_type.transform([test['transaction_type']])[0]
    test_df['merchant_encoded'] = le_merchant.transform([test['merchant_category']])[0]
    
    # Predict
    X_test_sample = test_df[features].values
    X_test_sample_scaled = scaler.transform(X_test_sample)
    prob = model.predict_proba(X_test_sample_scaled)[0][1]
    pred = "FRAUD" if prob >= best_threshold else "LEGIT"
    
    print(f"\n   {test['name']}:")
    print(f"   Amount: ${test['amount']}, Hour: {test['hour']}")
    print(f"   Fraud Probability: {prob*100:.1f}%")
    print(f"   Prediction: {pred}")