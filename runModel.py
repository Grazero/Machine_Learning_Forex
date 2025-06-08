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



@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # รับ JSON data
    # สมมติ client ส่งมาเป็น dict ของ feature
    features = pd.DataFrame([data])  # แปลงเป็น DataFrame
    
    proba = model.predict_proba(features)[:, 1][0]
    pred = int(proba > 0.4)
    
    return jsonify({'prediction': pred, 'probability': proba})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
