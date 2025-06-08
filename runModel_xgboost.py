from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# โหลดโมเดลที่เทรนไว้
model = joblib.load('rf_model_xgboost.pkl')

@app.route('/test', methods=['POST'])
def test():
    data = request.json
    print("✅ Received from MT5:", data)
    return jsonify({"status": "ok", "received": data})


@app.route('/')
def home():
    return "It works!"

expected_features = [
    'RSI', 'MACD', 'MACD_Diff',
    'BB_Upper', 'BB_Lower', 'BB_Width',
    'MA10', 'MA20', 'Close_lag2', 'Close_lag3',
    'Price_Change', 'Price_Change_Pct',
    'Candle_Range', 'Volume_Change', 'Volume_Spike',
    'MACD_RSI_Interaction', 'Price_MACD_Interaction'
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # รับ JSON data
    # สมมติ client ส่งมาเป็น dict ของ feature

    missing = [f for f in expected_features if f not in data]
    if missing:
        return jsonify({'error': f'Missing features: {missing}'}), 400

    features_df = pd.DataFrame([[data[f] for f in expected_features]], columns=expected_features)
    
    # ทำนายผล
    proba = model.predict_proba(features_df)[:, 1][0]
    pred = int(proba > 0.4)
    
    return jsonify({'prediction': pred, 'probability': float(proba)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
