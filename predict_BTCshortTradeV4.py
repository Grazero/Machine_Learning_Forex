# FILE NAME: predict_BTCshortTradeV3.py
import pandas as pd
import numpy as np
import json
import os
import sys
import joblib # สำหรับบันทึก/โหลด scaler และ model
from datetime import datetime
import time # <-- เพิ่มการ import โมดูล 'time' เข้ามา

# --- Global Configurations ---
# Path ไปยังโฟลเดอร์ MQL5\Files ของ MT5 Tester Agent (ต้องแก้ไขให้ตรงกับของคุณ!)
# !!! สำคัญ: เส้นทางนี้จะเปลี่ยนแปลงไปตามแต่ละครั้งที่รัน Strategy Tester
# คุณต้องอัปเดตเส้นทางนี้ในสคริปต์หากเส้นทางของ MT5 Tester Agent ของคุณเปลี่ยนไป
FIXED_AGENT_FILES_PATH = "C:\\Users\\graze\\AppData\\Roaming\\MetaQuotes\\Tester\\53785E099C927DB68A545C249CDBCE06\\Agent-127.0.0.1-3000\\MQL5\\Files" # <-- อัปเดตเส้นทางนี้!

INPUT_DATA_FILE_NAME = "input_data.json"
OUTPUT_RESULT_FILE_NAME = "prediction_result.txt"

# สร้าง Full Path สำหรับไฟล์ Input และ Output
input_file_path = os.path.join(FIXED_AGENT_FILES_PATH, INPUT_DATA_FILE_NAME)
output_file_path = os.path.join(FIXED_AGENT_FILES_PATH, OUTPUT_RESULT_FILE_NAME)

# Path ไปยังโมเดล, Scaler, และ Features List ที่เทรนไว้
MODEL_PATH = 'xgboost_model_v6_no_symbol_features.pkl'
SCALER_PATH = 'scaler_v6_no_symbol_features.pkl'
FEATURES_LIST_PATH = 'features_list_v6_no_symbol_features.pkl' # This file should contain the list of features used during training

# ตัวแปรสำหรับเก็บเวลาการแก้ไขไฟล์ Input ล่าสุด
LAST_INPUT_FILE_MOD_TIME = 0

# --- โหลดโมเดล, Scaler, และ Features List ที่เทรนไว้ ---
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features_list = joblib.load(FEATURES_LIST_PATH) # Load the exact features list used in training
    print(f"✅ โมเดล XGBoost '{MODEL_PATH}' โหลดสำเร็จ.")
    print(f"✅ Scaler '{SCALER_PATH}' โหลดสำเร็จ.")
    print(f"✅ Features list '{FEATURES_LIST_PATH}' โหลดสำเร็จ. จำนวน Features: {len(features_list)}")
except FileNotFoundError as e:
    print(f"❌ ข้อผิดพลาด: ไม่พบไฟล์ที่จำเป็น - {e}. โปรดตรวจสอบว่าไฟล์โมเดล, scaler, และ features list อยู่ในไดเรกทอรีเดียวกันกับสคริปต์นี้.")
    sys.exit() # หยุดการทำงานหากไม่พบไฟล์
except Exception as e:
    print(f"❌ ข้อผิดพลาดในการโหลดโมเดลหรือส่วนประกอบ: {e}")
    sys.exit(1) # หยุดการทำงานหากมีข้อผิดพลาดอื่น


# --- ฟังก์ชันหลักในการเตรียมข้อมูลสำหรับทำนาย (รับ Features ที่คำนวณแล้ว) ---
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

# --- Main File Monitoring Loop ---
def main_file_monitor():
    global LAST_INPUT_FILE_MOD_TIME

    print(f"Python Prediction Service Started. Monitoring for '{input_file_path}'...")

    # ตรวจสอบว่าเส้นทางที่กำหนดมีอยู่จริงหรือไม่
    if not os.path.isdir(FIXED_AGENT_FILES_PATH):
        print(f"⚠️ คำเตือน: เส้นทาง FIXED_AGENT_FILES_PATH '{FIXED_AGENT_FILES_PATH}' ไม่มีอยู่จริง.")
        print("โปรดตรวจสอบให้แน่ใจว่าไดเรกทอรีนี้มีอยู่จริงและเป็นเส้นทางที่ถูกต้องสำหรับ MT5 Tester Agent ปัจจุบันของคุณ.")
        print("สคริปต์จะยังคงทำงานต่อไป แต่อาจไม่พบไฟล์.")

    # Initialize features_data_json outside the loop to ensure it's always defined
    features_data_json = "" 

    while True:
        try:
            # ตรวจสอบว่า input_data.json มีอยู่หรือไม่
            if os.path.exists(input_file_path):
                current_modified_time = os.path.getmtime(input_file_path)
                
                # ตรวจสอบว่าไฟล์ถูกแก้ไขหรือไม่ (ข้อมูลใหม่ถูกเขียนโดย EA)
                if current_modified_time > LAST_INPUT_FILE_MOD_TIME:
                    # อ่าน input_data.json ด้วย encoding UTF-8
                    with open(input_file_path, 'r', encoding='utf-8') as f:
                        features_data_json = f.read()
                    
                    LAST_INPUT_FILE_MOD_TIME = current_modified_time # อัปเดตเวลาการแก้ไขล่าสุด
                    
                    print(f"📖 ได้รับ Features จาก MT5 (ผ่านไฟล์ '{INPUT_DATA_FILE_NAME}')...") 

                    # แปลง JSON string เป็น Python dictionary
                    features_dict = json.loads(features_data_json)

                    # เตรียมข้อมูลสำหรับทำนาย
                    try:
                        input_features_df = prepare_data_for_prediction(features_dict, features_list)
                        print(f"DEBUG: Input features DataFrame shape for prediction: {input_features_df.shape}")
                        # print(f"DEBUG: Input features DataFrame columns: {input_features_df.columns.tolist()}") # Uncomment for detailed debug

                        # ทำนายผล
                        # ใช้ prediction_threshold เดียวกันกับที่ใช้ในการประเมินผลการเทรน
                        prediction_threshold = 0.40 
                        
                        y_pred_proba = model.predict_proba(input_features_df)[:, 1] # ความน่าจะเป็นของ Class 1
                        pred_class = (y_pred_proba > prediction_threshold).astype(int)[0] # ทำนายตามเกณฑ์ที่กำหนด
                        
                        result = {'prediction': int(pred_class), 'probability': float(y_pred_proba[0])}
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
                        print(f"❌ ข้อผิดพลาดในการลบไฟล์ input: {e}")
            
            time.sleep(0.01) # ตรวจสอบทุก 10ms เพื่อการตอบสนองที่รวดเร็ว

        except json.JSONDecodeError as e:
            print(f"❌ ข้อผิดพลาดในการถอดรหัส JSON จาก '{INPUT_DATA_FILE_NAME}': {e}. เนื้อหา: '{features_data_json[:200]}...'")
            if os.path.exists(input_file_path):
                try:
                    os.remove(input_file_path)
                    print(f"🗑️ ลบไฟล์ '{INPUT_DATA_FILE_NAME}' ที่เสียหาย.")
                except Exception as e_clean:
                    print(f"❌ ไม่สามารถลบไฟล์ input ที่เสียหายได้: {e_clean}")
        except Exception as e:
            print(f"❌ ข้อผิดพลาดที่ไม่คาดคิดเกิดขึ้นใน Main Loop: {e}")
            if os.path.exists(input_file_path):
                try:
                    os.remove(input_file_path)
                    print(f"🗑️ ลบไฟล์ '{INPUT_DATA_FILE_NAME}' เนื่องจากข้อผิดพลาดที่ไม่คาดคิด.")
                except Exception as e_clean:
                    print(f"❌ ไม่สามารถลบไฟล์ input หลังจากข้อผิดพลาดที่ไม่คาดคิดได้: {e_clean}")


if __name__ == '__main__':
    main_file_monitor() # เริ่มต้นการตรวจสอบไฟล์ทันที
