# FILE NAME: predict_LSTM_API.py
# Python Flask API สำหรับทำนายผลด้วยโมเดล LSTM และสื่อสารกับ MT5 ผ่าน HTTP API

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model # ต้องใช้ load_model จาก tensorflow.keras.models

app = Flask(__name__)

# --- การตั้งค่า (Configuration) ---
# ตรวจสอบให้แน่ใจว่า EXPORT_DIR นี้ตรงกับที่ตั้งที่คุณบันทึกโมเดล, scaler และ features_list
# เปลี่ยน EXPORT_DIR เป็นสตริงว่าง เนื่องจากไฟล์อยู่ในไดเรกทอรีปัจจุบันแล้ว
EXPORT_DIR = '' 

# กำหนดค่า LOOKBACK_PERIOD ให้ตรงกับที่ใช้ในการฝึกโมเดล LSTM
# ควรโหลดจาก features_list.pkl หรือกำหนดให้ตรงกัน
# หากคุณไม่แน่ใจว่าค่านี้ควรเป็นเท่าไหร่ ให้ตรวจสอบใน LSTM_ForexTraderScriptV2.py
# หรือใน features_list.pkl ที่ถูกสร้างขึ้นมา
LOOKBACK_PERIOD = 60 # Default value, ensure this matches your training script

# โหลดโมเดล, scaler และ features_list เมื่อเริ่มต้นแอปพลิเคชัน
# การโหลดที่นี่จะทำให้ไม่ต้องโหลดซ้ำทุกครั้งที่มี request เข้ามา
try:
    # แก้ไขชื่อไฟล์ให้เป็น .h5 เพื่อให้เป็นไฟล์เดี่ยวที่ Keras สามารถโหลดได้ง่าย
    model_path = os.path.join(EXPORT_DIR, 'lstm_forex_model_combined.keras') # เปลี่ยนเป็น .h5
    scaler_path = os.path.join(EXPORT_DIR, 'lstm_forex_model_combined_scaler.pkl')
    features_list_path = os.path.join(EXPORT_DIR, 'lstm_forex_model_combined_features_list.pkl')

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    features_list = joblib.load(features_list_path)

    # ตรวจสอบว่า LOOKBACK_PERIOD ที่ใช้ในโมเดลตรงกับที่คาดไว้
    # โดยปกติ LSTM layer จะมี input_shape เป็น (timesteps, features)
    # เราสามารถดึง timesteps จาก model.input_shape[1]
    if model.input_shape and len(model.input_shape) > 1:
        LOOKBACK_PERIOD = model.input_shape[1]
        print(f"✅ Loaded model with LOOKBACK_PERIOD: {LOOKBACK_PERIOD}")
    else:
        print(f"⚠️ Could not determine LOOKBACK_PERIOD from model input shape. Using default: {LOOKBACK_PERIOD}")

    print("✅ โมเดล, Scaler และ Feature List ถูกโหลดสำเร็จแล้ว")
    print(f"Features expected by model: {features_list}")

except Exception as e:
    print(f"❌ ข้อผิดพลาดในการโหลดโมเดลหรือไฟล์ที่เกี่ยวข้อง: {e}")
    print("โปรดตรวจสอบว่าไฟล์ 'lstm_forex_model_combined.h5', 'lstm_forex_model_combined_scaler.pkl', และ 'lstm_forex_model_combined_features_list.pkl' อยู่ในโฟลเดอร์ที่รันสคริปต์นี้ และชื่อไฟล์ถูกต้อง")
    # ออกจากโปรแกรมหากโหลดโมเดลไม่ได้
    exit()

@app.route('/')
def home():
    """
    หน้าแรกของ API สำหรับตรวจสอบว่า API ทำงานอยู่หรือไม่
    """
    return "LSTM Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint สำหรับรับข้อมูล Features จาก MT5 และส่งคืนสัญญาณการเทรด
    """
    try:
        data = request.json # รับ JSON data ที่ส่งมาจาก MT5
        print(f"✅ ได้รับข้อมูลจาก MT5: {len(data)} แท่งเทียน")

        if not isinstance(data, list) or not data:
            return jsonify({'error': 'Input data must be a non-empty list of feature dictionaries.'}), 400

        # แปลงข้อมูล JSON ที่ได้รับให้เป็น DataFrame
        # MT5 ส่งข้อมูลมาเป็น list ของ dicts โดยแต่ละ dict คือ 1 แท่งเทียน
        # เราต้องเรียงลำดับ feature ให้ตรงกับ features_list ที่โมเดลคาดหวัง
        
        # สร้าง DataFrame จากข้อมูลที่ได้รับ
        input_df = pd.DataFrame(data)
        
        # ตรวจสอบว่า features ที่ได้รับครบถ้วนและเรียงลำดับถูกต้อง
        # หากไม่ครบหรือมี feature เกิน อาจทำให้เกิดปัญหาได้
        received_features = input_df.columns.tolist()
        if set(features_list) != set(received_features):
            missing_features = set(features_list) - set(received_features)
            extra_features = set(received_features) - set(features_list)
            error_message = "Feature mismatch:"
            if missing_features:
                error_message += f" Missing: {list(missing_features)}."
            if extra_features:
                error_message += f" Extra: {list(extra_features)}."
            print(f"❌ {error_message}")
            return jsonify({'error': error_message}), 400

        # จัดเรียงคอลัมน์ของ DataFrame ให้ตรงกับ features_list ที่โมเดลถูกฝึกมา
        input_df = input_df[features_list]

        # ตรวจสอบและจัดการค่า NaN/Inf (ถ้ามี) ก่อน Scale
        input_df = input_df.replace([np.inf, -np.inf], np.nan).fillna(0) # แทนที่ inf ด้วย NaN แล้วเติม NaN ด้วย 0

        # การ Scale ข้อมูล
        # scaler.transform คาดหวัง 2D array (samples, features)
        scaled_data = scaler.transform(input_df)

        # Reshape ข้อมูลสำหรับ LSTM: (samples, timesteps, features)
        # ในกรณีนี้ samples คือ 1 (เพราะเราส่งข้อมูลมาทีละ lookback_period แท่ง)
        # timesteps คือ LOOKBACK_PERIOD
        # features คือ จำนวน features ใน features_list
        num_features = len(features_list)
        
        # ตรวจสอบว่าจำนวนแท่งเทียนที่ได้รับตรงกับ LOOKBACK_PERIOD
        if scaled_data.shape[0] != LOOKBACK_PERIOD:
            error_msg = f"จำนวนแท่งเทียนที่ได้รับ ({scaled_data.shape[0]}) ไม่ตรงกับ LOOKBACK_PERIOD ({LOOKBACK_PERIOD}) ที่โมเดลคาดหวัง"
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 400

        reshaped_data = scaled_data.reshape(1, LOOKBACK_PERIOD, num_features)

        # ทำนายผล
        # model.predict_proba() ใช้สำหรับโมเดลที่ให้ความน่าจะเป็น (เช่น Logistic Regression, RandomForest)
        # สำหรับ Keras/TensorFlow LSTM model, ใช้ model.predict() ซึ่งจะคืนค่าความน่าจะเป็นโดยตรง
        prediction_proba = model.predict(reshaped_data)[0][0] # ได้ค่าความน่าจะเป็นของคลาส 1 (Buy Signal)

        # แปลงความน่าจะเป็นเป็นสัญญาณ BUY/SELL/NEUTRAL
        # คุณอาจต้องปรับ Threshold นี้ตามผล Backtest ของคุณ
        BUY_THRESHOLD = 0.55 # ตัวอย่าง: ถ้าความน่าจะเป็น Buy > 55%
        SELL_THRESHOLD = 0.45 # ตัวอย่าง: ถ้าความน่าจะเป็น Sell < 45% (หมายถึงความน่าจะเป็น Buy ต่ำ)

        signal = "NEUTRAL"
        if prediction_proba > BUY_THRESHOLD:
            signal = "BUY"
        elif prediction_proba < SELL_THRESHOLD:
            signal = "SELL"

        print(f"✅ ทำนายผลสำเร็จ: Probability={prediction_proba:.4f}, Signal={signal}")
        return jsonify({'signal': signal, 'probability': float(prediction_proba)}), 200

    except KeyError as e:
        print(f"❌ ข้อผิดพลาด: Feature '{e}' ไม่พบในข้อมูลที่ส่งมา")
        return jsonify({'error': f"Missing feature in input data: {e}"}), 400
    except Exception as e:
        print(f"❌ ข้อผิดพลาดที่ไม่คาดคิด: {e}")
        return jsonify({'error': f"An unexpected error occurred: {e}"}), 500

if __name__ == '__main__':
    # รัน Flask app
    # host='0.0.0.0' ทำให้สามารถเข้าถึงได้จากภายนอก (เช่น MT5 ที่รันบนเครื่องเดียวกัน)
    # port=5000 คือพอร์ตที่ API จะเปิด
    print("🚀 กำลังเริ่มต้น Flask API บน http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000)
