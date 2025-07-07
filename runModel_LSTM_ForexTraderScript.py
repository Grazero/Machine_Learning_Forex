# FILE NAME: predict_LSTMTrade.py
# สคริปต์ Python สำหรับทำนายผลด้วยโมเดล LSTM และสื่อสารกับ MT5 ผ่านไฟล์

import pandas as pd
import numpy as np
import json
import os
import sys
import joblib 
from datetime import datetime
import time 

# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import load_model 

# --- การตั้งค่า (Configuration) ---
# !!! สำคัญ: อัปเดตเส้นทางนี้ให้ตรงกับโฟลเดอร์ MQL5/Files ของ MT5 Tester Agent ของคุณ !!!
# ตัวอย่าง: C:\Users\graze\AppData\Roaming\MetaQuotes\Tester\53785E099C927DB68A545C249CDBCE06\Agent-127.0.0.1-3000\MQL5\Files
FIXED_AGENT_FILES_PATH = "C:\\Users\\graze\\AppData\\Roaming\\MetaQuotes\\Tester\\53785E099C927DB68A545C249CDBCE06\\Agent-127.0.0.1-3000\\MQL5\\Files" 

INPUT_DATA_FILE_NAME = "input_data.json"
OUTPUT_RESULT_FILE_NAME = "prediction_result.txt"

# สร้าง Full Path สำหรับไฟล์ Input และ Output
input_file_path = os.path.join(FIXED_AGENT_FILES_PATH, INPUT_DATA_FILE_NAME)
output_file_path = os.path.join(FIXED_AGENT_FILES_PATH, OUTPUT_RESULT_FILE_NAME)

# Path ไปยังโมเดล, Scaler, และ Features List ที่เทรนไว้
# !!! สำคัญ: ต้องเป็นไฟล์ที่มาจากโมเดล LSTM ของคุณที่เทรนไว้ก่อนหน้านี้
MODEL_PATH = './models/lstm_forextrader_modelV1.keras' # อัปเดตชื่อไฟล์โมเดล
SCALER_PATH = './scaler_lstmV1.pkl' # อัปเดตชื่อไฟล์ Scaler
FEATURES_LIST_PATH = './features_list_lstmV1.pkl' # อัปเดตชื่อไฟล์ Features List
CLASS_WEIGHTS_PATH = './class_weights_lstmV1.pkl' # เพิ่ม Class Weights Path

# --- ฟังก์ชันโหลดโมเดลและ Scaler ---
def load_resources():
    """โหลดโมเดล LSTM, MinMaxScaler และ Features List ที่บันทึกไว้"""
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        features_list = joblib.load(FEATURES_LIST_PATH)
        
        # โหลด class_weights (อาจไม่ได้ใช้โดยตรงในการทำนาย แต่เก็บไว้เพื่อความสมบูรณ์)
        class_weights = joblib.load(CLASS_WEIGHTS_PATH)
        
        print("✅ โหลดโมเดล, Scaler, Features List และ Class Weights สำเร็จแล้ว")
        return model, scaler, features_list, class_weights
    except Exception as e:
        print(f"❌ ข้อผิดพลาดในการโหลดทรัพยากร: {e}")
        print(f"โปรดตรวจสอบว่าไฟล์เหล่านี้มีอยู่จริง:\n- {MODEL_PATH}\n- {SCALER_PATH}\n- {FEATURES_LIST_PATH}\n- {CLASS_WEIGHTS_PATH}")
        sys.exit(1) # ออกจากโปรแกรมหากโหลดทรัพยากรไม่ได้

# --- ฟังก์ชันทำนายสัญญาณ ---
def predict_signal(model, scaler, features_list, input_data_df, lookback_period=60):
    """
    รับ DataFrame ข้อมูล Features ล่าสุด (Lookback_Period แท่ง)
    และใช้โมเดล LSTM ในการทำนาย
    """
    # ตรวจสอบว่า input_data_df มีคอลัมน์ที่จำเป็นครบถ้วนหรือไม่
    missing_features = [f for f in features_list if f not in input_data_df.columns]
    if missing_features:
        print(f"❌ ข้อผิดพลาด: ข้อมูล Input ขาด Features ที่จำเป็น: {missing_features}")
        return "ERROR: Missing Features"

    # เลือกเฉพาะคอลัมน์ Features ที่ใช้ในการเทรนโมเดล
    # และแปลงเป็น numpy array สำหรับ scaler
    features_raw = input_data_df[features_list].values
    
    # Scale ข้อมูล Features
    # เนื่องจาก scaler ถูก fit ด้วยข้อมูล 2D (samples, features)
    # เราจึงต้องรักษารูปร่าง 2D นี้ไว้
    features_scaled = scaler.transform(features_raw)

    # ปรับรูปร่างข้อมูลให้เหมาะสมกับ LSTM (Samples, Timesteps, Features)
    # ในกรณีนี้ เราคาดว่า input_data_df มี Lookback_Period แท่งอยู่แล้ว
    # ดังนั้น samples = 1, timesteps = lookback_period, features = จำนวน features
    # features_scaled.shape ควรเป็น (lookback_period, num_features)
    # เราต้องการให้เป็น (1, lookback_period, num_features)
    
    if features_scaled.shape[0] != lookback_period:
        print(f"❌ ข้อผิดพลาด: จำนวนแท่งเทียนใน Input ({features_scaled.shape[0]}) ไม่ตรงกับ LOOKBACK_PERIOD ({lookback_period})")
        return "ERROR: Invalid Input Length"

    X_predict = features_scaled.reshape(1, lookback_period, features_scaled.shape[1])

    # ทำนายด้วยโมเดล (ผลลัพธ์เป็น probabilities สำหรับแต่ละคลาส)
    raw_prediction = model.predict(X_predict) # จะได้ array เช่น [[0.1, 0.8, 0.1]]

    # หา Class ที่มี Probability สูงสุด
    predicted_class = np.argmax(raw_prediction, axis=1)[0] # ได้ 0, 1, หรือ 2

    # แปลงผลลัพธ์ตัวเลขเป็นข้อความสัญญาณ
    # 0: Sell, 1: Buy, 2: Neutral
    if predicted_class == 1:
        signal = "BUY"
    elif predicted_class == 0:
        signal = "SELL"
    else: # predicted_class == 2
        signal = "NEUTRAL"
        
    print(f"ทำนาย: Raw probabilities: {raw_prediction[0]}, Predicted Class: {predicted_class} -> Signal: {signal}")
    return signal

# --- Main Loop สำหรับการสื่อสารกับ EA ---
def main():
    # โหลดทรัพยากรทั้งหมดเมื่อเริ่มต้นเพียงครั้งเดียว
    model, scaler, features_list, class_weights = load_resources()

    print(f"✅ สคริปต์ Python สำหรับทำนายพร้อมทำงานแล้ว. กำลังรอไฟล์ Input ที่: {input_file_path}")

    last_checked_time = 0

    while True:
        try:
            # ตรวจสอบว่ามีไฟล์ input_data.json หรือไม่
            if os.path.exists(input_file_path):
                # ตรวจสอบเวลาที่ไฟล์ถูกแก้ไขล่าสุด
                current_file_mtime = os.path.getmtime(input_file_path)

                if current_file_mtime > last_checked_time:
                    # อ่านข้อมูลจากไฟล์ JSON
                    with open(input_file_path, 'r', encoding='utf-8') as f:
                        features_data_json = f.read()
                    
                    # แปลง JSON เป็น DataFrame
                    features_data_list = json.loads(features_data_json)
                    features_data_df = pd.DataFrame(features_data_list)
                    
                    # แปลงคอลัมน์ 'time' เป็น datetime และตั้งเป็น index
                    features_data_df['time'] = pd.to_datetime(features_data_df['time'], unit='s') # สมมติว่า EA ส่ง timestamp มาเป็นวินาที
                    features_data_df = features_data_df.set_index('time')

                    # ทำนายสัญญาณ
                    # LOOKBACK_PERIOD ควรจะเท่ากับที่ใช้ในการเทรนโมเดล (เช่น 60)
                    # ซึ่ง EA จะต้องส่งข้อมูลมา 60 แท่งเทียนพอดี
                    predicted_signal = predict_signal(model, scaler, features_list, features_data_df, lookback_period=60)

                    # เขียนผลลัพธ์ไปยังไฟล์ output_result.txt
                    with open(output_file_path, 'w', encoding='utf-8') as f:
                        f.write(predicted_signal)
                    
                    print(f"🚀 ทำนายและเขียนผลลัพธ์ '{predicted_signal}' ไปยัง '{OUTPUT_RESULT_FILE_NAME}' แล้ว")

                    # อัปเดตเวลาที่ตรวจสอบไฟล์ล่าสุด
                    last_checked_time = current_file_mtime
                    
                    # ลบไฟล์ Input เพื่อให้ EA สามารถเขียนไฟล์ใหม่ได้
                    try:
                        os.remove(input_file_path)
                        print(f"🗑️ ลบไฟล์ '{INPUT_DATA_FILE_NAME}' แล้ว")
                    except Exception as e:
                        print(f"❌ ข้อผิดพลาดในการลบไฟล์ input: {e}. อาจถูกล็อกโดยกระบวนการอื่น.")
            
            time.sleep(0.01) # ตรวจสอบทุก 10ms เพื่อการตอบสนองที่รวดเร็ว

        except json.JSONDecodeError as e:
            # กรณีที่ไฟล์ JSON เสียหาย (อาจถูกเขียนไม่สมบูรณ์)
            print(f"❌ ข้อผิดพลาดในการถอดรหัส JSON จาก '{INPUT_DATA_FILE_NAME}': {e}. เนื้อหา: '{features_data_json[:200]}...'")
            # if os.path.exists(input_file_path):
            #     try:
            #         #os.remove(input_file_path)
            #         print(f"🗑️ ลบไฟล์ '{INPUT_DATA_FILE_NAME}' ที่เสียหาย เพื่อให้ EA เขียนใหม่.")
            #     except Exception as e_clean:
            #         print(f"❌ ไม่สามารถลบไฟล์ input ที่เสียหายได้: {e_clean}")
            time.sleep(0.1) # พักนานขึ้นหากมีปัญหาการอ่านไฟล์
        except Exception as e:
            # ข้อผิดพลาดที่ไม่คาดคิดอื่น ๆ ใน Main Loop
            print(f"❌ ข้อผิดพลาดที่ไม่คาดคิดเกิดขึ้นใน Main Loop: {e}")
            if os.path.exists(input_file_path):
                try:
                    os.remove(input_file_path)
                    print(f"🗑️ ลบไฟล์ '{INPUT_DATA_FILE_NAME}' เนื่องจากข้อผิดพลาดที่ไม่คาดคิด เพื่อให้ EA เขียนใหม่.")
                except Exception as e_clean:
                    print(f"❌ ไม่สามารถลบไฟล์ input ที่เสียหายได้: {e_clean}")
            time.sleep(1) # พักนานขึ้นหากมีข้อผิดพลาดที่ไม่คาดคิด


if __name__ == "__main__":
    main()
