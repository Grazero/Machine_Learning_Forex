from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# และใน Flask:

model = joblib.load('xgboost_shortBTC_term_model.pkl')

@app.route('/test', methods=['POST'])
def test():
    data = request.json
    print("✅ Received from MT5:", data)
    return jsonify({"status": "ok", "received": data})


@app.route('/')
def home():
    return "It works!"

expected_features = [
    'open', 'high', 'low', 'close', 'tick_volume', # ราคาและ Volume พื้นฐาน
    'RSI', 'Stoch_K', 'Stoch_D', # Momentum indicators
    'EMA_fast', 'EMA_slow', 'EMA_diff', # Trend indicators
    'MACD', 'MACD_signal', 'MACD_hist', # MACD indicators
    'ATR', # Volatility
    'Price_change', 'Body_size', 'Upper_shadow', 'Lower_shadow', # Price Action
    'Upper_shadow_ratio', 'Lower_shadow_ratio', # Price Action Ratios
    'return_1', 'return_2', 'return_3', # Quick price change (เพิ่ม return_2, return_3)
    'RSI_lag1', 'MACD_hist_lag1', 'ATR_lag1', # Lagged key indicators
    'Stoch_K_lag1', 'Stoch_D_lag1', 'EMA_fast_lag1', 'EMA_slow_lag1', 'close_lag1', # Lagged other indicators
    'volume_avg', # Smoothed Volume
    # เพิ่มฟีเจอร์ Divergence ใหม่
    'bullish_rsi_divergence',
    'bearish_rsi_divergence',
    # เพิ่มฟีเจอร์แนวโน้มและการกลับตัวเพิ่มเติม
    'BB_width', 'BB_percent', # Bollinger Bands
    'EMA_cross_signal', # EMA Crossover
    'RSI_ROC', 'MACD_hist_ROC', # Rate of Change
    'ADX', 'ADX_pos', 'ADX_neg', # ADX indicators
    'Price_change_ATR_ratio', # Volatility-adjusted Price Change
    # เพิ่มฟีเจอร์ Candlestick Patterns ใหม่
    'bullish_engulfing',
    'bearish_engulfing',
    'hammer',
    'shooting_star'
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # รับ JSON data
    # สมมติ client ส่งมาเป็น dict ของ feature
    print("✅ Received from MT5:", data)
    # ตรวจสอบว่ามีฟีเจอร์ที่จำเป็นครบหรือไม่
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
