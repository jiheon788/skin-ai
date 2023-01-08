from ast import keyword
from email import header
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request
import os

options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = webdriver.Chrome(options=options)
driver.get("https://www.google.co.kr/imghp?hl=ko&ogbl")

# ftype = "jpg"
searchWord = "건선"
stopPoint = 5

elem = driver.find_element_by_name("q")
elem.send_keys(f"{searchWord}")
elem.send_keys(Keys.RETURN)



# SCROLL_PAUSE_TIME = 1

# # Get scroll height
# last_height = driver.execute_script("return document.body.scrollHeight")

# while True:
#     # Scroll down to bottom
#     driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

#     # Wait to load page
#     time.sleep(SCROLL_PAUSE_TIME)

#     # Calculate new scroll height and compare with last scroll height
#     new_height = driver.execute_script("return document.body.scrollHeight")
#     if new_height == last_height:
#       try:
#         driver.find_element_by_css_selector(".mye4qd").click()
#       except:
#         break
#     last_height = new_height


images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")
count = 1

try:
  os.mkdir(f"output/{searchWord}")
except:
  pass

for image in images:
  try:
    image.click()
    time.sleep(2)
    imgUrl = driver.find_element_by_xpath("/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div/a/img").get_attribute("src")
    opener=urllib.request.build_opener()
    opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(imgUrl, f"output/{searchWord}/{searchWord}{count}.jpg")
    # urllib.request.urlretrieve(imgUrl, f"{searchWord}{count}.{ftype}")
    count += 1
    if count == stopPoint:
      break
  except:
    pass

driver.close()