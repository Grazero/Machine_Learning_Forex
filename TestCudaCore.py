import torch  r33432432 
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")

print(torch.__version__)           # ดูเวอร์ชัน
print(torch.version.cuda)          # ดูเวอร์ชัน CUDA ที่ PyTorch รองรับ
print(torch.cuda.is_available())   # True = เห็น GPU
