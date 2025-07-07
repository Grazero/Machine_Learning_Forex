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

# --- Global Configurations ---
# Path ไปยังโฟลเดอร์ MQL5\Files ของ MT5 Tester Agent หรือ MT5 Terminal
# !!! สำคัญ: คุณต้องแก้ไขเส้นทางนี้ให้ตรงกับสภาพแวดล้อมของคุณ
# สำหรับ Strategy Tester จะเป็นเส้นทางที่คล้ายกับ:
# "C:\\Users\\<YourUser>\\AppData\\Roaming\\MetaQuotes\\Tester\\<AgentID>\\MQL5\\Files"
# สำหรับการใช้งานจริงบน MT5 Terminal ควรเป็น:
# "C:\\Users\\<YourUser>\\AppData\\Roaming\\MetaQuotes\\Terminal\\<YourMT5InstanceID>\\MQL5\\Files"
# แนะนำให้ตรวจสอบใน MT5 Terminal โดยไปที่ File -> Open Data Folder -> MQL5 -> Files
FIXED_AGENT_FILES_PATH = "C:\\Users\\graze\\AppData\\Roaming\\MetaQuotes\\Tester\\53785E099C927DB68A545C249CDBCE06\\Agent-127.0.0.1-3000\\MQL5\\Files" # <-- อัปเดตเส้นทางนี้!

INPUT_DATA_FILE_NAME = "input_data.json"
OUTPUT_RESULT_FILE_NAME = "prediction_result.txt"

# สร้าง Full Path สำหรับไฟล์ Input และ Output
input_file_path = os.path.join(FIXED_AGENT_FILES_PATH, INPUT_DATA_FILE_NAME)
output_file_path = os.path.join(FIXED_AGENT_FILES_PATH, OUTPUT_RESULT_FILE_NAME)

# Path ไปยังโมเดล, Scaler, และ Features List ที่เทรนไว้
# !!! สำคัญ: ต้องเป็นไฟล์ที่มาจากโมเดล LSTM ของคุณที่เทรนไว้ก่อนหน้านี้
MODEL_PATH = './models/best_lstm_model.keras' 
SCALER_PATH = './scalers/minmax_scaler.joblib'
FEATURES_LIST_PATH = './scalers/features_list.joblib'

# ตัวแปรสำหรับเก็บเวลาการแก้ไขไฟล์ Input ล่าสุด เพื่อตรวจสอบการเปลี่ยนแปลง
LAST_INPUT_FILE_MOD_TIME = 0

# กำหนด LSTM_TIMESTEPS ให้ตรงกับที่ใช้ในการเทรน
# !!! สำคัญ: ตัวเลขนี้ต้องตรงกับ LSTM_TIMESTEPS ในสคริปต์การเทรนของคุณ (เช่น 20)
LSTM_TIMESTEPS = 60 # <-- อัปเดตตัวเลขนี้ให้ตรง!

# --- โหลดโมเดล, Scaler, และ Features List ที่เทรนไว้ ---
try:
    model = load_model(MODEL_PATH) # ใช้ tf.keras.models.load_model สำหรับ Keras model
    scaler = joblib.load(SCALER_PATH)
    features_list = joblib.load(FEATURES_LIST_PATH) 
    print(f"✅ โมเดล LSTM '{MODEL_PATH}' โหลดสำเร็จ.")
    print(f"✅ Scaler '{SCALER_PATH}' โหลดสำเร็จ.")
    print(f"✅ Features list '{FEATURES_LIST_PATH}' โหลดสำเร็จ. จำนวน Features: {len(features_list)}")
    print(f"✅ กำหนด LSTM_TIMESTEPS: {LSTM_TIMESTEPS}")
except FileNotFoundError as e:  
    print(f"❌ ข้อผิดพลาด: ไม่พบไฟล์ที่จำเป็น - {e}. โปรดตรวจสอบว่าไฟล์โมเดล, scaler, และ features list อยู่ในไดเรกทอรีเดียวกันกับสคริปต์นี้ หรือระบุ PATH ที่ถูกต้อง.")
    sys.exit() # หยุดการทำงานหากไม่พบไฟล์ที่สำคัญ
except Exception as e:
    print(f"❌ ข้อผิดพลาดในการโหลดโมเดลหรือส่วนประกอบ: {e}")
    sys.exit(1) # หยุดการทำงานหากมีข้อผิดพลาดอื่น


# --- ฟังก์ชันหลักในการเตรียมข้อมูลสำหรับทำนาย (รับ Features ที่คำนวณแล้ว) ---
def prepare_data_for_prediction(features_data, expected_features_list, timesteps):
    """
    รับ dictionary ของ features ที่คำนวณแล้วจาก MT5 ซึ่งคาดว่ามีข้อมูลสำหรับหลาย Timesteps
    แปลงเป็น NumPy array, ตรวจสอบความถูกต้องของ features, และ Scale ข้อมูล.
    
    Args:
        features_data (dict): Dictionary ที่มี key เป็นชื่อ feature และ value เป็น list ของค่า feature
                              สำหรับ 'timesteps' ที่ผ่านมา.
                              เช่น {'RSI_H1': [val_t-N, ..., val_t], 'EMA_fast_H1': [val_t-N, ..., val_t], ...}
        expected_features_list (list): รายชื่อ features ที่คาดหวังตามลำดับที่ใช้ในการเทรน.
        timesteps (int): จำนวน timesteps ที่โมเดล LSTM คาดหวัง.
        
    Returns:
        np.ndarray: ข้อมูลที่ Scale แล้วในรูปแบบ (1, timesteps, num_features) พร้อมสำหรับการทำนาย.
    Raises:
        ValueError: หากข้อมูลไม่ครบถ้วนหรือไม่ถูกต้องตาม Timesteps หรือ Features.
    """
    
    # ตรวจสอบว่าทุก feature ใน expected_features_list มีอยู่ใน features_data
    # และแต่ละ feature มี list ของค่าที่มีความยาวถูกต้อง (เท่ากับ timesteps)
    for feature_name in expected_features_list:
        if feature_name not in features_data:
            raise ValueError(f"Feature '{feature_name}' ที่คาดหวังขาดหายไปในข้อมูลที่ได้รับจาก MT5.")
        
        if not isinstance(features_data[feature_name], list):
            raise ValueError(f"Feature '{feature_name}' ไม่ได้เป็น list. ได้รับ: {type(features_data[feature_name])}. โปรดตรวจสอบ JSON format จาก MT5.")
            
        if len(features_data[feature_name]) != timesteps:
            raise ValueError(f"Feature '{feature_name}' มีความยาวไม่เท่ากับ Timesteps ({timesteps}). ได้รับ: {len(features_data[feature_name])}. โปรดตรวจสอบการเก็บประวัติใน EA.")

    # สร้าง DataFrame จาก features_data
    # แถวคือ Timesteps, คอลัมน์คือ Features
    df_features = pd.DataFrame(features_data)

    # ตรวจสอบว่าคอลัมน์ใน df_features ตรงกับที่คาดหวัง และเรียงลำดับ
    missing_features_in_df = [f for f in expected_features_list if f not in df_features.columns]
    if missing_features_in_df:
        raise ValueError(f"Features ที่ขาดหายไปใน DataFrame หลังสร้าง: {missing_features_in_df}. ตรวจสอบ keys ใน JSON ที่ส่งมาจาก EA.")

    df_features_ordered = df_features[expected_features_list]

    # ตรวจสอบค่า NaN/inf ก่อน Scaling
    if df_features_ordered.isnull().any().any() or np.isinf(df_features_ordered).any().any():
        print(f"DEBUG: NaN or Inf found in raw features before scaling: NaNs={df_features_ordered.isnull().sum().sum()}, Infs={np.isinf(df_features_ordered).sum().sum()}")
        # สามารถเลือกที่จะเติมค่า NaN/Inf ได้ที่นี่ หาก EA ไม่สามารถจัดการได้
        # ตัวอย่าง: เติม NaN ด้วย 0, และ Inf ด้วยค่าที่จำกัด
        df_features_ordered = df_features_ordered.replace([np.inf, -np.inf], np.nan).fillna(0) 

    # Scale Features
    # scaler.transform คาดหวัง 2D array (samples, features)
    scaled_features_2d = scaler.transform(df_features_ordered)
    
    # Reshape สำหรับ LSTM: (samples, timesteps, num_features)
    # ในกรณีนี้, เรามี 1 sample (คือ 1 เหตุการณ์การทำนาย), timesteps จำนวน timesteps, และ num_features จำนวน features
    num_features = len(expected_features_list)
    scaled_features_lstm_input = scaled_features_2d.reshape(1, timesteps, num_features)
    
    return scaled_features_lstm_input

# --- Main File Monitoring Loop ---
def main_file_monitor():
    global LAST_INPUT_FILE_MOD_TIME

    print(f"Python LSTM Prediction Service Started. Monitoring for '{input_file_path}'...")

    if not os.path.isdir(FIXED_AGENT_FILES_PATH):
        print(f"⚠️ คำเตือน: เส้นทาง FIXED_AGENT_FILES_PATH '{FIXED_AGENT_FILES_PATH}' ไม่มีอยู่จริง.")
        print("โปรดตรวจสอบให้แน่ใจว่าไดเรกทอรีนี้มีอยู่จริงและเป็นเส้นทางที่ถูกต้องสำหรับ MT5 Terminal หรือ Tester Agent ปัจจุบันของคุณ.")
        print("สคริปต์จะยังคงทำงานต่อไป แต่อาจไม่พบไฟล์.")

    features_data_json = "" # Initialize outside loop to ensure it's always defined

    while True:
        try:
            if os.path.exists(input_file_path):
                current_modified_time = os.path.getmtime(input_file_path)
                
                if current_modified_time > LAST_INPUT_FILE_MOD_TIME:
                    # ใส่ delay เล็กน้อยเพื่อรอให้ MT5 เขียนไฟล์เสร็จสมบูรณ์
                    # สำคัญมากเพื่อหลีกเลี่ยงการอ่านไฟล์ที่ไม่สมบูรณ์
                    time.sleep(0.01) 
                    
                    with open(input_file_path, 'r', encoding='utf-8') as f:
                        features_data_json = f.read()
                    
                    LAST_INPUT_FILE_MOD_TIME = current_modified_time # อัปเดตเวลาการแก้ไขล่าสุด
                    
                    print(f"📖 ได้รับ Features จาก MT5 (ผ่านไฟล์ '{INPUT_DATA_FILE_NAME}')...") 

                    features_dict_received = json.loads(features_data_json)

                    try:
                        # เตรียมข้อมูลสำหรับทำนายสำหรับ LSTM
                        input_features_lstm = prepare_data_for_prediction(features_dict_received, features_list, LSTM_TIMESTEPS)
                        print(f"DEBUG: Input features LSTM shape for prediction: {input_features_lstm.shape}")

                        # ทำนายผลด้วยโมเดล LSTM
                        # output ของ model.predict() สำหรับ classification คือ probability ตรงๆ
                        y_pred_proba = model.predict(input_features_lstm).flatten()[0] # จะได้ค่า probability (0-1)

                        # ใช้ prediction_threshold เดียวกันกับที่ใช้ในการประเมินผลการเทรน
                        prediction_threshold = 0.50 # ตรวจสอบให้แน่ใจว่าค่านี้ตรงกับที่คุณใช้ในการเทรนและประเมินผล
                        
                        pred_class = int(y_pred_proba > prediction_threshold) # แปลงเป็น 1 (Buy) หรือ 0 (Sell/No Trade)
                        
                        result = {'prediction': pred_class, 'probability': float(y_pred_proba)}
                        print(f"⚡ ทำนายผลสำเร็จ: {result}")

                    except ValueError as ve:
                        result = {'prediction': -1, 'probability': 0.0, 'error': str(ve)}
                        print(f"❌ ข้อผิดพลาดในการเตรียมข้อมูลหรือทำนาย: {ve}")
                    except Exception as e:
                        result = {'prediction': -1, 'probability': 0.0, 'error': str(e)}
                        print(f"❌ ข้อผิดพลาดที่ไม่คาดคิดในการทำนาย: {e}")
                    
                    # เขียนผลการทำนายลงใน prediction_result.txt ด้วย encoding UTF-8
                    with open(output_file_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False) 
                    print(f"✍️ เขียนผลการทำนายลงใน '{OUTPUT_RESULT_FILE_NAME}': {result}")
                    
                    # ลบ input.json เพื่อส่งสัญญาณให้ EA ว่าประมวลผลเสร็จแล้วและสามารถเขียนข้อมูลใหม่ได้
                    try:
                        os.remove(input_file_path)
                        print(f"🗑️ ลบ '{INPUT_DATA_FILE_NAME}' เพื่อส่งสัญญาณว่าเสร็จสิ้น.")
                    except OSError as e:
                        print(f"❌ ข้อผิดพลาดในการลบไฟล์ input: {e}. อาจถูกล็อกโดยกระบวนการอื่น.")
            
            time.sleep(0.01) # ตรวจสอบทุก 10ms เพื่อการตอบสนองที่รวดเร็ว

        except json.JSONDecodeError as e:
            # กรณีที่ไฟล์ JSON เสียหาย (อาจถูกเขียนไม่สมบูรณ์)
            print(f"❌ ข้อผิดพลาดในการถอดรหัส JSON จาก '{INPUT_DATA_FILE_NAME}': {e}. เนื้อหา: '{features_data_json[:200]}...'")
            if os.path.exists(input_file_path):
                try:
                    os.remove(input_file_path)
                    print(f"🗑️ ลบไฟล์ '{INPUT_DATA_FILE_NAME}' ที่เสียหาย เพื่อให้ EA เขียนใหม่.")
                except Exception as e_clean:
                    print(f"❌ ไม่สามารถลบไฟล์ input ที่เสียหายได้: {e_clean}")
            time.sleep(0.1) # พักนานขึ้นหากมีปัญหาการอ่านไฟล์
        except Exception as e:
            # ข้อผิดพลาดที่ไม่คาดคิดอื่น ๆ ใน Main Loop
            print(f"❌ ข้อผิดพลาดที่ไม่คาดคิดเกิดขึ้นใน Main Loop: {e}")
            if os.path.exists(input_file_path):
                try:
                    os.remove(input_file_path)
                    print(f"🗑️ ลบไฟล์ '{INPUT_DATA_FILE_NAME}' เนื่องจากข้อผิดพลาดที่ไม่คาดคิด เพื่อให้ EA เขียนใหม่.")
                except Exception as e_clean:
                    print(f"❌ ไม่สามารถลบไฟล์ input หลังจากข้อผิดพลาดที่ไม่คาดคิดได้: {e_clean}")
            time.sleep(0.1) # พักนานขึ้นหากมีปัญหาทั่วไป

if __name__ == '__main__':
    main_file_monitor() 

