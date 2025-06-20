เช็ค Cuda เวอร์ชั่น 

nvidia-smi


python install version 3.10 
https://www.python.org/downloads/release/python-3100/

ติดตั้ง CUDA Toolkit 11.2:

คุณสามารถดาวน์โหลด CUDA Toolkit 11.2 ได้จาก Archive ของ NVIDIA: https://developer.nvidia.com/cuda-11-2-2-download-archive
เลือกเวอร์ชันที่ตรงกับระบบของคุณ (Windows, x86_64, 10 หรือ 11, exe (local)) และทำการติดตั้งตามปกติ
ดาวน์โหลดและตั้งค่า cuDNN SDK 8.1:

ไปที่เว็บไซต์ NVIDIA cuDNN Archive: https://developer.nvidia.com/rdp/cudnn-archive (คุณอาจต้องลงทะเบียนบัญชีนักพัฒนาของ NVIDIA ก่อน)
มองหา "Download cuDNN v8.1.1 (for CUDA 11.0, 11.1, 11.2)"
ดาวน์โหลดไฟล์ที่ตรงกับ Windows ของคุณ (เช่น "cuDNN v8.1.1 for CUDA 11.2")
แตกไฟล์ ZIP ที่ดาวน์โหลดมา คุณจะเห็นโฟลเดอร์ cuda
คัดลอกเนื้อหาทั้งหมดในโฟลเดอร์ cuda (ได้แก่ โฟลเดอร์ bin, include, lib) ไปยังไดเรกทอรีการติดตั้ง CUDA Toolkit 11.2 ของคุณ ซึ่งโดยปกติจะอยู่ที่ C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2
ตรวจสอบ PATH Environment Variable:

ตรวจสอบให้แน่ใจว่าได้เพิ่มพาธของ CUDA และ cuDNN ลงใน Environment Variables Path แล้ว
โดยทั่วไปพาธที่ควรจะมีคือ:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64
ถ้าคุณใช้ cuDNN 8.1.1 กับ CUDA 11.2 พาธเหล่านี้มักจะถูกตั้งค่าโดยอัตโนมัติจากการติดตั้ง CUDA และการคัดลอกไฟล์ cuDNN เข้าไปแล้ว


pip install --upgrade pip
pip install "tensorflow<2.11","pandas","joblib","ta","scikit-learn"


เช็คว่า tensorflow มองเห็น GPU มั้ย

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU Devices:", tf.config.list_physical_devices('GPU'))
