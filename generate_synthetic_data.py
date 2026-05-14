import pandas as pd
import numpy as np
import os

print("=" * 50)
print("GENERATING SYNTHETIC TRANSACTION DATA")
print("=" * 50)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

np.random.seed(42)
n_transactions = 50000

print(f"\n📊 Generating {n_transactions:,} transactions...")

# Create base data
data = {
    'transaction_id': range(1, n_transactions + 1),
    'amount': np.random.exponential(100, n_transactions),
    'hour': np.random.randint(0, 24, n_transactions).astype(int),
    'day_of_week': np.random.randint(0, 7, n_transactions).astype(int),
    'transaction_type': np.random.choice(['online', 'pos', 'atm', 'wire'], n_transactions),
    'merchant_category': np.random.choice(['retail', 'food', 'travel', 'entertainment', 'grocery'], n_transactions),
    'device_risk_score': np.random.uniform(0, 100, n_transactions),
    'ip_risk_score': np.random.uniform(0, 100, n_transactions),
    'is_foreign': np.random.choice([0, 1], n_transactions, p=[0.85, 0.15]),
}

df = pd.DataFrame(data)

# Inject fraud patterns (0.5% fraud rate)
fraud_count = int(n_transactions * 0.005)
fraud_indices = np.random.choice(n_transactions, fraud_count, replace=False)

print(f"💀 Injecting {fraud_count} fraudulent transactions...")

# Apply fraud patterns safely
df.loc[fraud_indices, 'amount'] = df.loc[fraud_indices, 'amount'] * np.random.uniform(3, 10, fraud_count)
df.loc[fraud_indices, 'device_risk_score'] = np.random.uniform(70, 100, fraud_count)
df.loc[fraud_indices, 'ip_risk_score'] = np.random.uniform(70, 100, fraud_count)
df.loc[fraud_indices, 'is_foreign'] = 1

# Fix: Convert hour values properly
unusual_hours = np.random.choice([0, 1, 2, 3, 4, 23], fraud_count)
for i, idx in enumerate(fraud_indices):
    df.at[idx, 'hour'] = int(unusual_hours[i])

df['is_fraud'] = 0
df.loc[fraud_indices, 'is_fraud'] = 1

# Save to data folder
csv_path = 'data/transactions.csv'
df.to_csv(csv_path, index=False)

print(f"\n✅ Dataset created successfully!")
print(f"   📁 Location: {csv_path}")
print(f"   📊 Total transactions: {len(df):,}")
print(f"   🚨 Fraud transactions: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
print(f"   ✅ Legit transactions: {len(df) - df['is_fraud'].sum():,}")

# Show first 5 rows
print(f"\n📋 First 5 transactions:")
print(df.head())

# Show fraud sample
print(f"\n📋 Sample of fraud transactions:")
print(df[df['is_fraud'] == 1][['amount', 'hour', 'device_risk_score', 'ip_risk_score', 'is_foreign']].head())