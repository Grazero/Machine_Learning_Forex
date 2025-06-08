from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Path ไปยังโมเดล, Scaler, และ Features List ที่เทรนไว้
MODEL_PATH = 'xgboost_model_v3.pkl'
SCALER_PATH = 'scaler_v3.pkl'
FEATURES_LIST_PATH = 'features_list_v3.pkl' # This file should contain the list of features used during training
# โหลดโมเดลที่เทรนไว้
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
features_list = joblib.load(FEATURES_LIST_PATH) # Load the exact features list used in training

def prepare_data_for_prediction(features_dict, expected_features_list):
    """
    รับ dictionary ของ features ที่คำนวณแล้วจาก MT5,
    แปลงเป็น DataFrame, ตรวจสอบความถูกต้องของ features, และ Scale ข้อมูล.
    """
    # สร้าง DataFrame จาก dictionary ที่ได้รับ (จะเป็น 1 แถว)
    df_features = pd.DataFrame([features_dict])

    # ตรวจสอบว่า features ที่ได้รับมาครบถ้วนและเรียงลำดับถูกต้องตามที่โมเดลคาดหวัง
    missing_features = [f for f in expected_features_list if f not in df_features.columns]
    if missing_features:
        raise ValueError(f"Features ที่ขาดหายไปในการทำนาย: {missing_features}. โปรดตรวจสอบ Feature Engineering ใน EA.")
    
    # เลือกและเรียงลำดับคอลัมน์ให้ตรงกับ features_list ที่ใช้ในการเทรน
    # นี่คือขั้นตอนสำคัญเพื่อให้แน่ใจว่าลำดับของ features ตรงกับที่ scaler และ model คาดหวัง
    df_features_ordered = df_features[expected_features_list]

    # Scale Features
    scaled_features = scaler.transform(df_features_ordered)
    
    # คืนค่าเป็น DataFrame (1 แถว) เพื่อให้ XGBoost รับได้
    return pd.DataFrame(scaled_features, columns=expected_features_list)

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

    missing = [f for f in features_list if f not in data]
    if missing:
        return jsonify({'error': f'Missing features: {missing}'}), 400

    
    input_features_df = prepare_data_for_prediction(data, features_list)

    # ทำนายผล
    # ใช้ prediction_threshold เดียวกันกับที่ใช้ในการประเมินผลการเทรน
    prediction_threshold = 0.40 
                            
    y_pred_proba = model.predict_proba(input_features_df)[:, 1] # ความน่าจะเป็นของ Class 1
    pred_class = (y_pred_proba > prediction_threshold).astype(int)[0] # ทำนายตามเกณฑ์ที่กำหนด
    
    result = {'prediction': int(pred_class), 'probability': float(y_pred_proba[0])}


    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
