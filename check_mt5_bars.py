import MetaTrader5 as mt5
from datetime import datetime, timedelta

# เชื่อมต่อ MT5
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()
    exit()

symbol = "BTCUSDm"
timeframes = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1
}
candles_to_check = 99999

# ทำให้แน่ใจว่า symbol ถูกเปิดใน Market Watch
if not mt5.symbol_select(symbol, True):
    print(f"ไม่สามารถเปิด {symbol}")
    mt5.shutdown()
    exit()

for tf_name, tf_code in timeframes.items():
    rates = mt5.copy_rates_from_pos(symbol, tf_code, 0, candles_to_check)
    if rates is None:
        print(f"[{tf_name}] ไม่สามารถโหลดข้อมูลได้")
    else:
        print(f"[{tf_name}] ดึงได้ {len(rates)} แท่ง")
        if len(rates) < candles_to_check:
            print(f"⚠️ มีแค่ {len(rates)} แท่งเท่านั้นใน {tf_name} (ขาด {candles_to_check - len(rates)})")

# ปิด MT5
mt5.shutdown()
