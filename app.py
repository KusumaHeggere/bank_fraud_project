from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load all model artifacts
print("=" * 50)
print("LOADING FRAUD DETECTION MODEL")
print("=" * 50)

try:
    model = joblib.load('models/fraud_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    le_transaction = joblib.load('models/label_encoder_transaction.pkl')
    le_merchant = joblib.load('models/label_encoder_merchant.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')
    print("✅ All model artifacts loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Please run 'python train_model.py' first")

def preprocess_transaction(transaction):
    """Convert raw transaction data to model-ready format"""
    
    # Create DataFrame
    df = pd.DataFrame([transaction])
    
    # Feature engineering (same as training)
    df['amount_log'] = np.log1p(df['amount'])
    df['combined_risk'] = (df['device_risk_score'] * 0.6 + df['ip_risk_score'] * 0.4) / 100
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_high_risk'] = (df['combined_risk'] > 0.7).astype(int)
    
    # Encode categorical variables
    try:
        df['transaction_type_encoded'] = le_transaction.transform([transaction['transaction_type']])[0]
    except ValueError:
        df['transaction_type_encoded'] = 0
    
    try:
        df['merchant_category_encoded'] = le_merchant.transform([transaction['merchant_category']])[0]
    except ValueError:
        df['merchant_category_encoded'] = 0
    
    # Select and order features
    X = df[feature_columns].values
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fraud Detection System</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 20px;
            }
            .container {
                max-width: 600px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }
            h1 {
                color: #667eea;
                text-align: center;
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                font-weight: bold;
                margin-bottom: 5px;
                color: #333;
            }
            input, select {
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
            }
            button {
                width: 100%;
                padding: 12px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                cursor: pointer;
                margin-top: 10px;
            }
            button:hover {
                transform: translateY(-2px);
            }
            .result {
                margin-top: 20px;
                padding: 15px;
                border-radius: 5px;
                text-align: center;
            }
            .fraud {
                background: #fee;
                border-left: 4px solid #c00;
                color: #c00;
            }
            .legit {
                background: #efe;
                border-left: 4px solid #080;
                color: #080;
            }
            .probability {
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🛡️ Bank Fraud Detection System</h1>
            <form method="POST" action="/predict">
                <div class="form-group">
                    <label>💰 Transaction Amount ($)</label>
                    <input type="number" step="0.01" name="amount" required>
                </div>
                <div class="form-group">
                    <label>⏰ Hour (0-23)</label>
                    <input type="number" name="hour" required min="0" max="23">
                </div>
                <div class="form-group">
                    <label>📅 Day of Week (0=Mon, 6=Sun)</label>
                    <input type="number" name="day_of_week" required min="0" max="6">
                </div>
                <div class="form-group">
                    <label>📱 Device Risk Score (0-100)</label>
                    <input type="number" step="0.1" name="device_risk_score" required>
                </div>
                <div class="form-group">
                    <label>🌐 IP Risk Score (0-100)</label>
                    <input type="number" step="0.1" name="ip_risk_score" required>
                </div>
                <div class="form-group">
                    <label>✈️ Foreign Transaction?</label>
                    <select name="is_foreign" required>
                        <option value="0">No (Domestic)</option>
                        <option value="1">Yes (Foreign)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>💳 Transaction Type</label>
                    <select name="transaction_type" required>
                        <option value="online">Online Purchase</option>
                        <option value="pos">Point of Sale</option>
                        <option value="atm">ATM Withdrawal</option>
                        <option value="wire">Wire Transfer</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>🏪 Merchant Category</label>
                    <select name="merchant_category" required>
                        <option value="retail">Retail</option>
                        <option value="food">Food & Dining</option>
                        <option value="travel">Travel</option>
                        <option value="entertainment">Entertainment</option>
                        <option value="grocery">Grocery</option>
                    </select>
                </div>
                <button type="submit">🔍 Analyze Transaction</button>
            </form>
        </div>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        transaction = {
            'amount': float(request.form['amount']),
            'hour': int(request.form['hour']),
            'day_of_week': int(request.form['day_of_week']),
            'device_risk_score': float(request.form['device_risk_score']),
            'ip_risk_score': float(request.form['ip_risk_score']),
            'is_foreign': int(request.form['is_foreign']),
            'transaction_type': request.form['transaction_type'],
            'merchant_category': request.form['merchant_category']
        }
        
        features = preprocess_transaction(transaction)
        prob = model.predict_proba(features)[0][1]
        pred = int(prob > 0.5)
        
        result_class = "fraud" if pred == 1 else "legit"
        result_title = "⚠️ FRAUD ALERT!" if pred == 1 else "✅ Transaction Legitimate"
        
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Result</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    max-width: 500px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 10px;
                    padding: 30px;
                    text-align: center;
                }}
                .result {{
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 10px;
                }}
                .fraud {{ background: #fee; color: #c00; }}
                .legit {{ background: #efe; color: #080; }}
                .probability {{ font-size: 48px; font-weight: bold; margin: 20px 0; }}
                a {{
                    display: inline-block;
                    margin-top: 20px;
                    color: #667eea;
                    text-decoration: none;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔍 Detection Result</h1>
                <div class="result {result_class}">
                    <h2>{result_title}</h2>
                    <div class="probability">Fraud Probability: {prob*100:.1f}%</div>
                </div>
                <a href="/">← Check Another Transaction</a>
            </div>
        </body>
        </html>
        '''
    except Exception as e:
        return f'<h1>Error: {str(e)}</h1><a href="/">Go Back</a>'

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🚀 STARTING FRAUD DETECTION WEB APP")
    print("=" * 50)
    print("\n👉 Open your browser and go to: http://localhost:5000")
    print("👉 Press CTRL+C to stop the server\n")
    app.run(debug=True, host='0.0.0.0', port=5000)