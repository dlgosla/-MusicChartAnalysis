import re, pickle
from selenium import webdriver
from singer import *

WAIT_TIME = 5

with open('singer_name.pickle', 'rb') as f:
    singer_name = pickle.load(f)

def GetMelonData():
    singers = []
    driver = webdriver.Chrome('chromedriver.exe')
    driver.implicitly_wait(WAIT_TIME)

    for name in singer_name[700:]:
        singer = Singer()
        singer.name = name
        name = name.replace('#', '%23')
        name = name.replace('&', '%26') 
        url = 'https://www.melon.com/search/total/index.htm?q='+ name + '&section=&searchGnbYn=Y&kkoSpl=Y&kkoDpType=&linkOrText=T&ipath=srch_form'
        driver.get(url)
        driver.implicitly_wait(WAIT_TIME)
        tmp = driver.find_elements_by_css_selector('#conts > div.section_atist > div > div.atist_dtl_info > dl > dd:nth-child(4)')[0].text
        if len(tmp) > 3:
            singer.sex, singer.group = tmp.split(',')
        else:
            singer.sex, singer.group = '.', '.'
        singer.group.strip()
        singer.fan = int(driver.find_elements_by_css_selector('#conts > div.section_atist > div > div.atist_dtl_info > div > span > span')[0].text.replace(',', ''))
        singers.append(singer)

    return singers

with open('singer.pickle', 'rb') as f:
    before = pickle.load(f)
print(len(before))

data = GetMelonData()


with open('singer.pickle', 'wb') as f:
    pickle.dump(before + data, f)

print("Done")