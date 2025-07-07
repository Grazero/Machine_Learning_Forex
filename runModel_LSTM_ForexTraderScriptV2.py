# FILE NAME: predict_LSTMTrade.py
# สคริปต์ Python สำหรับทำนายผลด้วยโมเดล LSTM และสื่อสารกับ MT5 Expert Advisor ผ่านไฟล์

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
# ตัวอย่าง: C:\\Users\\YourUser\\AppData\\Roaming\\MetaQuotes\\Tester\\<AgentID>\\MQL5\\Files
# หรือสำหรับบัญชีจริง: C:\\Users\\YourUser\\AppData\\Roaming\\MetaQuotes\\Terminal\\<TerminalID>\\MQL5\\Files
# ตรวจสอบเส้นทางที่ถูกต้องจาก MT5 Terminal -> File -> Open Data Folder -> MQL5 -> Files
FIXED_AGENT_FILES_PATH = "C:\\Users\\graze\\AppData\\Roaming\\MetaQuotes\\Tester\\53785E099C927DB68A545C249CDBCE06\\Agent-127.0.0.1-3000\\MQL5\\Files" 

INPUT_DATA_FILE_NAME = "input_data.json"
OUTPUT_RESULT_FILE_NAME = "prediction_result.txt"

# สร้าง Full Path สำหรับไฟล์ Input และ Output
input_file_path = os.path.join(FIXED_AGENT_FILES_PATH, INPUT_DATA_FILE_NAME)
output_file_path = os.path.join(FIXED_AGENT_FILES_PATH, OUTPUT_RESULT_FILE_NAME)

# Path ไปยังโมเดล, Scaler, และ Features List ที่เทรนไว้
# !!! สำคัญ: ต้องเป็นไฟล์ที่มาจากโมเดล LSTM ของคุณที่เทรนไว้ก่อนหน้านี้
# ตรวจสอบให้แน่ใจว่าชื่อไฟล์ตรงกับที่ export_model_and_params บันทึกไว้
MODEL_PATH = './exported_models/lstm_forex_model_combined.h5' 
SCALER_PATH = './exported_models/lstm_forex_model_combined_scaler.pkl' 
FEATURES_LIST_PATH = './exported_models/lstm_forex_model_combined_features_list.pkl' 

# LOOKBACK_PERIOD ควรจะเท่ากับที่ใช้ในการเทรนโมเดล (เช่น 60)
# ซึ่ง EA จะต้องส่งข้อมูลมา 60 แท่งเทียนพอดี
LOOKBACK_PERIOD_FOR_PREDICTION = 60 

# --- ฟังก์ชันโหลดโมเดลและ Scaler ---
def load_resources():
    """โหลดโมเดล LSTM, MinMaxScaler และ Features List ที่บันทึกไว้"""
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        features_list = joblib.load(FEATURES_LIST_PATH)
        
        print("✅ โหลดโมเดล, Scaler, และ Features List สำเร็จแล้ว")
        return model, scaler, features_list
    except Exception as e:
        print(f"❌ ข้อผิดพลาดในการโหลดทรัพยากร: {e}")
        print(f"โปรดตรวจสอบว่าไฟล์เหล่านี้มีอยู่จริง:\n- {MODEL_PATH}\n- {SCALER_PATH}\n- {FEATURES_LIST_PATH}")
        sys.exit(1) # ออกจากโปรแกรมหากโหลดทรัพยากรไม่ได้

# --- ฟังก์ชันทำนายสัญญาณ ---
def predict_signal(model, scaler, features_list, input_data_df, lookback_period):
    """
    รับ DataFrame ข้อมูล Features ล่าสุด (Lookback_Period แท่ง)
    และใช้โมเดล LSTM ในการทำนาย
    """
    # ตรวจสอบว่า input_data_df มีคอลัมน์ที่จำเป็นครบถ้วนหรือไม่
    missing_features = [f for f in features_list if f not in input_data_df.columns]
    if missing_features:
        print(f"❌ ข้อผิดพลาด: ข้อมูล Input ขาด Features ที่จำเป็น: {missing_features}")
        return "ERROR: Missing Features"

    # เลือกเฉพาะคอลัมน์ Features ที่ใช้ในการเทรนโมเดลและเรียงลำดับให้ถูกต้อง
    # การเรียงลำดับคอลัมน์ต้องตรงกับ features_list ที่ใช้ตอนเทรนโมเดล
    features_raw = input_data_df[features_list].values
    
    # Scale ข้อมูล Features ด้วย Scaler ที่โหลดมา
    # features_raw.shape ควรเป็น (lookback_period, num_features)
    features_scaled = scaler.transform(features_raw)

    # ปรับรูปร่างข้อมูลให้เหมาะสมกับ LSTM (Samples, Timesteps, Features)
    # ในกรณีนี้ เราคาดว่า input_data_df มี Lookback_Period แท่งอยู่แล้ว
    # ดังนั้น samples = 1, timesteps = lookback_period, features = จำนวน features
    
    if features_scaled.shape[0] != lookback_period:
        print(f"❌ ข้อผิดพลาด: จำนวนแท่งเทียนใน Input ({features_scaled.shape[0]}) ไม่ตรงกับ LOOKBACK_PERIOD ({lookback_period})")
        return "ERROR: Invalid Input Length"

    X_predict = features_scaled.reshape(1, lookback_period, features_scaled.shape[1])

    # ทำนายด้วยโมเดล (ผลลัพธ์เป็น probability ของคลาส 1 'ขึ้น')
    # โมเดลใช้ sigmoid activation ดังนั้น output จะเป็นค่าเดียวระหว่าง 0 ถึง 1
    raw_prediction = model.predict(X_predict, verbose=0).flatten()[0] # ได้ค่าเดียว เช่น 0.75

    # แปลงผลลัพธ์ตัวเลขเป็นข้อความสัญญาณ
    # Buy: ถ้าโมเดลทำนายว่ามีโอกาสขึ้นสูงกว่า 0.5 (หรือ Threshold ที่กำหนด)
    # Sell: ถ้าโมเดลทำนายว่ามีโอกาสขึ้นต่ำกว่า 0.5 (หรือ Threshold ที่กำหนด)
    # Neutral: อาจจะใช้สำหรับค่าที่อยู่ใกล้ 0.5 มากๆ หรือไม่มั่นใจ
    
    # สามารถปรับ Threshold ได้ตามความเหมาะสม
    PREDICTION_THRESHOLD_BUY = 0.51 # ต้องมั่นใจว่าขึ้นจริงๆ
    PREDICTION_THRESHOLD_SELL = 0.50 # ต้องมั่นใจว่าลงจริงๆ

    if raw_prediction >= PREDICTION_THRESHOLD_BUY:
        signal = "BUY"
    elif raw_prediction <= PREDICTION_THRESHOLD_SELL:
        signal = "SELL"
    else:
        signal = "NEUTRAL" # หากค่าอยู่ระหว่าง Thresholds หรือไม่มั่นใจ

    print(f"ทำนาย: Raw prediction (prob of UP): {raw_prediction:.4f} -> Signal: {signal}")
    return signal

# --- Main Loop สำหรับการสื่อสารกับ EA ---
def main():
    # โหลดทรัพยากรทั้งหมดเมื่อเริ่มต้นเพียงครั้งเดียว
    model, scaler, features_list = load_resources()

    print(f"✅ สคริปต์ Python สำหรับทำนายพร้อมทำงานแล้ว. กำลังรอไฟล์ Input ที่: {input_file_path}")

    last_checked_time = 0

    while True:
        # --- กำหนดค่าเริ่มต้นให้กับ predicted_signal ในแต่ละรอบของลูป ---
        predicted_signal = "NEUTRAL" # กำหนดค่าเริ่มต้น
        features_data_json = "" # กำหนดค่าเริ่มต้น
        # --- สิ้นสุดการกำหนดค่าเริ่มต้น ---

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
                    # EA ควรส่งข้อมูลมาในรูปแบบ List of Dictionaries
                    features_data_list = json.loads(features_data_json)
                    features_data_df = pd.DataFrame(features_data_list)
                    
                    # แปลงคอลัมน์ 'time' เป็น datetime และตั้งเป็น index
                    # สมมติว่า EA ส่ง timestamp มาเป็นวินาที (Unix timestamp)
                    # ถ้า EA ส่งมาในรูปแบบอื่น ต้องปรับตรงนี้
                    if 'time' in features_data_df.columns:
                        features_data_df['time'] = pd.to_datetime(features_data_df['time'], unit='s') 
                        features_data_df = features_data_df.set_index('time')
                    else:
                        print("❌ ข้อผิดพลาด: ไม่พบคอลัมน์ 'time' ในข้อมูล Input จาก EA.")
                        predicted_signal = "ERROR: No Time Column"
                        # หากไม่มีคอลัมน์ 'time' อาจจะใช้ RangeIndex ชั่วคราว หรือจัดการตามความเหมาะสม
                        # แต่ควรแจ้งให้ EA ส่งคอลัมน์ 'time' มาด้วย
                        
                    # ทำนายสัญญาณ
                    # LOOKBACK_PERIOD_FOR_PREDICTION ควรจะเท่ากับที่ใช้ในการเทรนโมเดล (เช่น 60)
                    # ซึ่ง EA จะต้องส่งข้อมูลมา LOOKBACK_PERIOD_FOR_PREDICTION แท่งเทียนพอดี
                    # ตรวจสอบเงื่อนไขที่นี่เพื่อให้แน่ใจว่า predicted_signal มีค่าที่ถูกต้องก่อนที่จะใช้
                    if not predicted_signal.startswith("ERROR"): # ตรวจสอบว่าไม่ใช่ค่า ERROR ที่ตั้งไว้ก่อนหน้า
                        predicted_signal = predict_signal(model, scaler, features_list, features_data_df, lookback_period=LOOKBACK_PERIOD_FOR_PREDICTION)

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
            # ในกรณีที่ไฟล์เสียหาย อาจจะลองลบไฟล์เพื่อให้ EA เขียนใหม่
            if os.path.exists(input_file_path):
                try:
                    #os.remove(input_file_path)
                    print(f"🗑️ ลบไฟล์ '{INPUT_DATA_FILE_NAME}' ที่เสียหาย เพื่อให้ EA เขียนใหม่.")
                except Exception as e_clean:
                    print(f"❌ ไม่สามารถลบไฟล์ input ที่เสียหายได้: {e_clean}")
            time.sleep(0.1) # พักนานขึ้นหากมีปัญหาการอ่านไฟล์
        except Exception as e:
            # ข้อผิดพลาดที่ไม่คาดคิดอื่น ๆ ใน Main Loop
            print(f"❌ ข้อผิดพลาดที่ไม่คาดคิดเกิดขึ้นใน Main Loop: {e}")
            # พิมพ์ค่าของ predicted_signal เพื่อช่วยในการดีบั๊ก
            print(f"DEBUG: Value of predicted_signal before error: {predicted_signal}")
            if os.path.exists(input_file_path):
                try:
                    #os.remove(input_file_path)
                    print(f"🗑️ ลบไฟล์ '{INPUT_DATA_FILE_NAME}' เนื่องจากข้อผิดพลาดที่ไม่คาดคิด เพื่อให้ EA เขียนใหม่.")
                except Exception as e_clean:
                    print(f"❌ ไม่สามารถลบไฟล์ input ที่เสียหายได้: {e_clean}")
            time.sleep(1) # พักนานขึ้นหากมีข้อผิดพลาดที่ไม่คาดคิด


if __name__ == "__main__":
    main()
