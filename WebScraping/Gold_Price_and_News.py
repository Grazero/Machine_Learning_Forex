# FILE NAME: gold_bot.py
# Python Web Scraping Bot สำหรับดึงราคาทองคำและข่าวสาร

import requests
from bs4 import BeautifulSoup
import time # สำหรับการหน่วงเวลา (rate limiting)
import re   # สำหรับ Regular Expressions (ถ้าจำเป็น)

# --- เพิ่ม: สำหรับ Selenium ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


def get_gold_price():
    """
    ดึงราคาทองคำวันนี้จาก goldprice.org โดยใช้ Selenium
    เพื่อจัดการกับเนื้อหาที่โหลดด้วย JavaScript และค้นหาตาม class 'gpoticker-price'
    """
    url = "https://goldprice.org/" 
    
    # --- ตั้งค่า Selenium WebDriver ---
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--ignore-certificate-errors") # อาจช่วยเรื่อง SSL
    chrome_options.add_argument("--allow-insecure-localhost") # อาจช่วยเรื่อง SSL สำหรับ localhost
    chrome_options.add_argument("--disable-blink-features=AutomationControlled") # ซ่อนร่องรอยการเป็น Bot
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"]) # ซ่อนร่องรอยการเป็น Bot
    chrome_options.add_argument(f"user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    # เพิ่ม service_args เพื่อจัดการ SSL errors ที่ระดับ ChromeDriver
    service_args = ['--ignore-ssl-errors=true', '--ssl-protocol=any']

    driver = None
    print(f"กำลังดึงราคาทองคำจาก: {url} (ใช้ Selenium)")
    try:
        # ส่ง service_args ไปยัง Service
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install(), service_args=service_args), options=chrome_options)
        driver.set_page_load_timeout(30)
        driver.get(url)
        
        # รอให้ JavaScript โหลดข้อมูลและ element ที่ต้องการปรากฏ
        # รอจนกว่า p ที่มี class="hidden" และข้อความ "THB" ปรากฏ
        # ซึ่งบ่งชี้ว่าส่วนของราคาทองคำบาทไทยได้โหลดแล้ว
        WebDriverWait(driver, 20).until( # เพิ่ม timeout เป็น 20 วินาที
            EC.presence_of_element_located((By.XPATH, "//p[@class='hidden' and text()='THB']"))
        )
        time.sleep(3) # เพิ่มเวลา sleep อีกเล็กน้อยเพื่อให้มั่นใจว่าข้อมูลโหลดครบถ้วน

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        gold_prices = {}
        
        # ค้นหา div ที่มี class 'container' และ 'row'
        container_div = soup.find('div', class_='container')
        if container_div:
            row_div = container_div.find('div', class_='row')
            if row_div:
                # ค้นหา div ที่มี p class="hidden" และ text เป็น "THB"
                thb_price_section = row_div.find('p', string='THB')
                if thb_price_section:
                    # เมื่อเจอส่วน THB, ให้หา parent ของมัน (ซึ่งคือ div ที่ครอบ ticker-value price-value)
                    # แล้วค่อยหา span.gpoticker-price ที่อยู่ภายใน
                    
                    # วิธีที่แม่นยำที่สุดคือการหา div.gpoticker-row ที่มี label "ราคาทองคำ 96.5% (บาทละ) ซื้อ" และ "ขาย"
                    # จากการตรวจสอบล่าสุดบน goldprice.org/gold-price-thailand.html
                    # ราคาทองคำบาทละยังคงอยู่ใน div.gpoticker-row และมี label ที่ชัดเจน
                    price_rows = row_div.find_all('div', class_='gpoticker-row')
                    for row in price_rows:
                        label_element = row.find('div', class_='gpoticker-label')
                        price_span = row.find('span', class_='gpoticker-price')
                        
                        if label_element and price_span:
                            label_text = label_element.text.strip()
                            price_text = price_span.text.strip()
                            
                            if "ราคาทองคำ 96.5% (บาทละ) ซื้อ" in label_text:
                                gold_prices['ราคาทองคำแท่ง 96.5% (บาทละ) ซื้อ'] = price_text
                            elif "ราคาทองคำ 96.5% (บาทละ) ขาย" in label_text:
                                gold_prices['ราคาทองคำแท่ง 96.5% (บาทละ) ขาย'] = price_text
                                
        if gold_prices:
            return gold_prices
        else:
            print("ไม่พบข้อมูลราคาทองคำที่ต้องการบนหน้าเว็บ (อาจมีการเปลี่ยนแปลงโครงสร้าง HTML หรือข้อมูลยังไม่โหลด)")
            return None

    except TimeoutException:
        print(f"❌ ข้อผิดพลาด: โหลดหน้าเว็บ {url} หมดเวลา.")
        return None
    except WebDriverException as e:
        print(f"❌ ข้อผิดพลาด WebDriver (Selenium): {e}")
        print("โปรดตรวจสอบว่า Chrome Browser ถูกติดตั้งแล้ว และ ChromeDriver ทำงานได้ถูกต้อง")
        print("หากรันบน Linux Server อาจต้องติดตั้ง Chromium: sudo apt install -y chromium-browser")
        return None
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการแยกวิเคราะห์ข้อมูลราคาทองคำ: {e}")
        return None
    finally:
        if driver:
            driver.quit()

def get_gold_news(num_articles=5):
    """
    ดึงหัวข้อข่าวทองคำจาก Investing.com (เวอร์ชันภาษาไทย) โดยใช้ Selenium
    """
    url = "https://th.investing.com/commodities/gold-news"
    
    # --- ตั้งค่า Selenium WebDriver ---
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--ignore-certificate-errors") # อาจช่วยเรื่อง SSL
    chrome_options.add_argument(f"user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    driver = None
    print(f"\nกำลังดึงข่าวทองคำจาก: {url} (ใช้ Selenium)")
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        driver.set_page_load_timeout(30)
        driver.get(url)
        
        # รอให้ JavaScript โหลดข้อมูลและ element ที่ต้องการปรากฏ
        # รอจนกว่า div ที่มี class 'textDiv' ปรากฏ
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "textDiv"))
        )
        time.sleep(3) # เพิ่มเวลา sleep อีกเล็กน้อยเพื่อให้มั่นใจว่าข้อมูลโหลดครบถ้วน

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        news_articles = []
        news_containers = soup.find_all('div', class_='textDiv')

        for i, container in enumerate(news_containers):
            if i >= num_articles:
                break
            
            title_element = container.find('a', class_='title')
            
            if title_element:
                title = title_element.text.strip()
                link = "https://th.investing.com" + title_element['href']
                news_articles.append({"title": title, "link": link})

        if news_articles:
            return news_articles
        else:
            print("ไม่พบข่าวทองคำที่ต้องการบนหน้าเว็บ")
            return None

    except TimeoutException:
        print(f"❌ ข้อผิดพลาด: โหลดหน้าเว็บ {url} หมดเวลา.")
        return None
    except WebDriverException as e:
        print(f"❌ ข้อผิดพลาด WebDriver (Selenium): {e}")
        print("โปรดตรวจสอบว่า Chrome Browser ถูกติดตั้งแล้ว และ ChromeDriver ทำงานได้ถูกต้อง")
        print("หากรันบน Linux Server อาจต้องติดตั้ง Chromium: sudo apt install -y chromium-browser")
        return None
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการแยกวิเคราะห์ข้อมูลข่าวทองคำ: {e}")
        return None
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    # --- ดึงราคาทองคำ ---
    gold_price_data = get_gold_price()
    if gold_price_data:
        print("\n--- ราคาทองคำวันนี้ ---")
        for key, value in gold_price_data.items():
            print(f"{key}: {value}")
    else:
        print("\nไม่สามารถดึงราคาทองคำได้ในขณะนี้")

    # หน่วงเวลาเล็กน้อยเพื่อไม่ให้ส่ง Request ถี่เกินไป (ป้องกันการถูกบล็อก)
    time.sleep(2) 

    # --- ดึงข่าวทองคำ ---
    gold_news_data = get_gold_news(num_articles=5)
    if gold_news_data:
        print("\n--- ข่าวทองคำล่าสุด ---")
        for i, article in enumerate(gold_news_data):
            print(f"{i+1}. หัวข้อ: {article['title']}")
            print(f"   ลิงก์: {article['link']}")
    else:
        print("\nไม่สามารถดึงข่าวทองคำได้ในขณะนี้")
