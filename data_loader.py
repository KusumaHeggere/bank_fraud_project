import pandas as pd
import numpy as np
import os

def load_real_dataset(file_path):
    """Load real Kaggle credit card fraud dataset"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded real dataset: {len(df)} transactions")
        print(f"   Fraud rate: {df['Class'].mean()*100:.4f}%")
        return df
    except Exception as e:
        print(f"❌ Could not load real dataset: {e}")
        return None

def load_synthetic_dataset():
    """Generate synthetic dataset"""
    np.random.seed(42)
    n_transactions = 100000  # Increased to 100k
    
    # Generate features matching required format
    df = pd.DataFrame({
        'amount': np.random.exponential(100, n_transactions),
        'hour': np.random.randint(0, 24, n_transactions),
        'day_of_week': np.random.randint(0, 7, n_transactions),
        'transaction_type': np.random.choice(['online', 'pos', 'atm', 'wire'], n_transactions),
        'merchant_category': np.random.choice(['retail', 'food', 'travel', 'entertainment', 'grocery'], n_transactions),
        'device_risk_score': np.random.uniform(0, 100, n_transactions),
        'ip_risk_score': np.random.uniform(0, 100, n_transactions),
        'is_foreign': np.random.choice([0, 1], n_transactions, p=[0.85, 0.15]),
    })
    
    # Inject fraud (0.5%)
    fraud_count = int(n_transactions * 0.005)
    fraud_idx = np.random.choice(n_transactions, fraud_count, replace=False)
    df.loc[fraud_idx, 'amount'] *= np.random.uniform(3, 10, fraud_count)
    df.loc[fraud_idx, 'device_risk_score'] = np.random.uniform(70, 100, fraud_count)
    df.loc[fraud_idx, 'ip_risk_score'] = np.random.uniform(70, 100, fraud_count)
    df.loc[fraud_idx, 'is_foreign'] = 1
    df['is_fraud'] = 0
    df.loc[fraud_idx, 'is_fraud'] = 1
    
    return df

# Use based on what's available
if os.path.exists('data/creditcard.csv'):
    df = load_real_dataset('data/creditcard.csv')
else:
    print("No real dataset found. Generating synthetic data...")
    df = load_synthetic_dataset()
    df.to_csv('data/transactions.csv', index=False)