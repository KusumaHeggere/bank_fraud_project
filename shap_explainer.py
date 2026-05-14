import shap
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

print("=" * 50)
print("SHAP MODEL EXPLAINABILITY")
print("=" * 50)

# Load model and data
model = joblib.load('models/xgboost_fraud_model.pkl')
scaler = joblib.load('models/scaler_xgb.pkl')
features = joblib.load('models/features_xgb.pkl')

df = pd.read_csv('data/transactions.csv')

# Prepare sample data
sample_df = df[features].head(100)
sample_scaled = scaler.transform(sample_df)

# Create SHAP explainer
print("📊 Creating SHAP explainer...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample_scaled)

# Summary plot
print("📈 Generating SHAP summary plot...")
shap.summary_plot(shap_values, sample_scaled, feature_names=features, show=False)
plt.savefig('models/shap_summary.png', bbox_inches='tight', dpi=150)
print("✅ Saved: models/shap_summary.png")

# Feature importance
print("\n🔝 Top 5 most important features:")
importance_df = pd.DataFrame({
    'feature': features,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

for i, row in importance_df.head().iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# Force plot for a single prediction (for API)
def explain_prediction(transaction_data):
    """Return SHAP explanation for a single transaction"""
    # Preprocess
    processed = scaler.transform(transaction_data.reshape(1, -1))
    shap_value = explainer.shap_values(processed)
    
    # Create explanation dict
    explanation = {}
    for i, feature in enumerate(features):
        explanation[feature] = float(shap_value[0][i])
    
    return explanation

print("\n✅ SHAP explainer ready for API integration!")