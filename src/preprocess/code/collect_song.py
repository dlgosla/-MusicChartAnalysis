import re, pickle
from selenium import webdriver
from song import *

WAIT_TIME = 5
YEAR = '2021년'
#MONTH = ['01월', '02월']
#MONTH = ['03월', '04월']
MONTH = ['05월', '06월']
#MONTH = ['07월', '08월', '09월']
#MONTH = ['10월', '11월', '12월']

def GetMelonData():

      songs = []
      driver = webdriver.Chrome('chromedriver.exe')
      driver.implicitly_wait(WAIT_TIME)

      url = 'https://www.melon.com/chart/index.htm'
      driver.get(url)

      # 차트 파인더 클릭
      
      chartFinder = driver.find_element_by_xpath("//*[@id='gnb_menu']/ul[1]/li[1]/div/div/button")
      chartFinder.click()
      driver.implicitly_wait(WAIT_TIME)

      # 차트선택 class='on'으로 변경
      activatingSelection = driver.find_element_by_xpath("//*[@id='d_chart_search']/h3")
      driver.execute_script("arguments[0].setAttribute('class', 'on')", activatingSelection)
      # 월간차트 class='on' 변경 및 클릭
      monthlyChart = driver.find_element_by_class_name("tab02")
      driver.execute_script("arguments[0].setAttribute('class', 'tab02 on')", monthlyChart)
      monthlyChart.click()
      driver.implicitly_wait(WAIT_TIME)

      activatingEraSelection = driver.find_element_by_xpath("//*[@id='d_chart_search']/div/div/div[1]")
      driver.execute_script("arguments[0].setAttribute('class', 'box_chic nth1 view on')", activatingEraSelection)
      eraList = driver.find_elements_by_xpath("//*[@id='d_chart_search']/div/div/div[1]/div[1]/ul/li")
      eraIgnore = ['1990년대', '1980년대']
      isGnr2 = True

      # 연대 활성화 및 선택
      for era in eraList:
            if era.text in eraIgnore:
                  continue
            driver.execute_script("arguments[0].setAttribute('class', 'on')", era)
            era.click()
            #driver.execute_script("arguments[0].setAttribute('class', '')", era)
            driver.implicitly_wait(WAIT_TIME)

            #연도 활성화 및 선택
            activatingYearSelection = driver.find_element_by_xpath("//*[@id='d_chart_search']/div/div/div[2]")
            driver.execute_script("arguments[0].setAttribute('class', 'box_chic nth2 view on')", activatingYearSelection)
            yearList = driver.find_elements_by_xpath("//*[@id='d_chart_search']/div/div/div[2]/div[1]/ul/li")
            for year in yearList:
                  print(year.text)

                  if year.text != YEAR :
                        continue

                  driver.execute_script("arguments[0].setAttribute('class', 'on')", year)
                  year.click()
                  driver.implicitly_wait(WAIT_TIME)

                  #월간 활성화 및 선택
                  activatingMonthSelection = driver.find_element_by_xpath("//*[@id='d_chart_search']/div/div/div[3]")
                  driver.execute_script("arguments[0].setAttribute('class', 'box_chic nth3 view on')", activatingMonthSelection)
                  monthList = driver.find_elements_by_xpath("//*[@id='d_chart_search']/div/div/div[3]/div[1]/ul/li")

                  for month in monthList:
                        print(month.text)
                        if month.text not in MONTH:
                              continue
                        driver.execute_script("arguments[0].setAttribute('class', 'on')", month)
                        month.click()
                        driver.implicitly_wait(WAIT_TIME)

                        # 장르 활성화 및 선택
                        """
                        #1980s~2004/10: 국내종합 단일 (gnr_1)
                        #2004/11~2016/12: 가요 선택 (gnr_2)
                        #2017~: 국내종합 선택(gnr_2)
                        """
                        activatingGenreSelection = driver.find_element_by_xpath("//*[@id='d_chart_search']/div/div/div[5]")
                        driver.execute_script("arguments[0].setAttribute('class', 'box_chic last view on')", activatingGenreSelection)
                        if isGnr2 == True:
                              genreSelect = driver.find_element_by_id("gnr_2")
                              if year.text == '2004년' and month.text == '11월':
                                    isGnr2 = False
                        else:
                              genreSelect = driver.find_element_by_id("gnr_1")
                        driver.execute_script("arguments[0].setAttribute('class', 'on')", genreSelect)
                        genreSelect.click()
                        driver.implicitly_wait(WAIT_TIME)

                        # 검색
                        driver.find_element_by_xpath("//*[@id='d_srch_form']/div[2]/button").click()
                        driver.implicitly_wait(WAIT_TIME)
                        
                        objs = driver.find_elements_by_css_selector('#lst50 > td:nth-child(4) > div > a')
                        # top 1 ~ 50 긁기
                        for item in range(0, len(objs)):
                              song = Song()
                              song.year = int(year.text[:4])
                              song.month = int(month.text[:-1])
                              song.rank = item + 1
                              href = driver.find_elements_by_css_selector('#lst50 > td:nth-child(4) > div > a')[item].get_attribute('href')
                              number = re.findall('\d+', href)[0]
                              driver.execute_script("window.open('https://www.melon.com/song/detail.htm?songId="+number+"');")
                              driver.switch_to.window(driver.window_handles[1])
                              driver.implicitly_wait(WAIT_TIME)
                              song.title = driver.find_elements_by_css_selector('#downloadfrm > div > div > div.entry > div.info > div.song_name')[0].text
                              song.singer = driver.find_elements_by_css_selector('#downloadfrm > div > div > div.entry > div.info > div.artist > a > span:nth-child(1)')[0].text
                              song.album = driver.find_elements_by_css_selector('#downloadfrm > div > div > div.entry > div.meta > dl > dd:nth-child(2) > a')[0].text
                              song.genre = driver.find_elements_by_css_selector('#downloadfrm > div > div > div.entry > div.meta > dl > dd:nth-child(6)')[0].text
                              song.date = driver.find_elements_by_css_selector('#downloadfrm > div > div > div.entry > div.meta > dl > dd:nth-child(4)')[0].text
                              song.likes = int(driver.find_elements_by_css_selector('#d_like_count')[0].text.replace(',', ''))
                              lyrics = driver.find_elements_by_css_selector('#d_video_summary')
                              if len(lyrics) > 0:
                                    song.lyrics = lyrics[0].text
                              image_url = driver.find_elements_by_css_selector('#downloadfrm > div > div > div.thumb > a > img')[0].get_attribute('src')
                              delete_idx = image_url.find('?')
                              song.setImage(image_url[:delete_idx])
                              driver.close()
                              driver.switch_to.window(driver.window_handles[0])
                              driver.implicitly_wait(WAIT_TIME)
                              songs.append(song)
                              print(song.rank, song.title)
                        
                        # 51 ~ 100 긁기
                        objs = driver.find_elements_by_css_selector('#lst100 > td:nth-child(4) > div > a')
                        for item in range(0, len(objs)):
                              song = Song()
                              song.year = int(year.text[:4])
                              song.month = int(month.text[:-1])
                              song.rank = item + 51
                              href = driver.find_elements_by_css_selector('#lst100 > td:nth-child(4) > div > a')[item].get_attribute('href')
                              number = re.findall('\d+', href)[0]
                              driver.execute_script("window.open('https://www.melon.com/song/detail.htm?songId="+number+"');")
                              driver.switch_to.window(driver.window_handles[1])
                              driver.implicitly_wait(WAIT_TIME)
                              song.title = driver.find_elements_by_css_selector('#downloadfrm > div > div > div.entry > div.info > div.song_name')[0].text
                              song.singer = driver.find_elements_by_css_selector('#downloadfrm > div > div > div.entry > div.info > div.artist > a > span:nth-child(1)')[0].text
                              song.album = driver.find_elements_by_css_selector('#downloadfrm > div > div > div.entry > div.meta > dl > dd:nth-child(2) > a')[0].text
                              song.genre = driver.find_elements_by_css_selector('#downloadfrm > div > div > div.entry > div.meta > dl > dd:nth-child(6)')[0].text
                              song.date = driver.find_elements_by_css_selector('#downloadfrm > div > div > div.entry > div.meta > dl > dd:nth-child(4)')[0].text
                              song.likes = int(driver.find_elements_by_css_selector('#d_like_count')[0].text.replace(',', ''))
                              lyrics = driver.find_elements_by_css_selector('#d_video_summary')
                              if len(lyrics) > 0:
                                    song.lyrics = lyrics[0].text
                              image_url = driver.find_elements_by_css_selector('#downloadfrm > div > div > div.thumb > a > img')[0].get_attribute('src')
                              delete_idx = image_url.find('?')
                              song.setImage(image_url[:delete_idx])
                              driver.close()
                              driver.switch_to.window(driver.window_handles[0])
                              driver.implicitly_wait(WAIT_TIME)
                              songs.append(song)
                              print(song.rank, song.title)
                        
                  break
      return songs

data = GetMelonData()

with open('data.pickle', 'rb') as f:
      before = pickle.load(f)

with open('data.pickle', 'wb') as f:
      pickle.dump(before + data, f)

