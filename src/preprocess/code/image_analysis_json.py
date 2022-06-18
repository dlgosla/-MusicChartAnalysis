import os
import cv2
import numpy as np
import math

import re
import pickle
import requests

import json

from collections import OrderedDict

from io import BytesIO

from PIL import Image



#-------------------------------------------------

class Album:
    def __init__(self):
        self.album = ''
        self.published_date = ''

        self.title = []
        self.genre = []

class Song2:
      def __init__(self):
            self.title = 0
            self.singer = 0
            self.album = 0
            self.genre = 0
            self.date = 0
            self.likes = 0

#-------------------------------------------------

val_Y = 100
val_X = 100

path_dir = './images'

file_list = os.listdir(path_dir)

#이미지와 이미지 이름을 list로 넣는다.
img_list = []
img_name_list = []


for i in range(len(file_list)):
    img_name = path_dir + '/'+ file_list[i]
    img_array = np.fromfile(img_name, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    imp_title = file_list[i].replace('.jpg','')

    #크기 변형
    img_list.append(cv2.resize(img, dsize=(100,100),interpolation=cv2.INTER_AREA))
    
    img_name_list.append(imp_title)

#이미지 압축 완료


#=======================================================
#색의 편차 수준을 구하는 함수
#이미지의 복잡한 정도를 반영한다.
#고주파 성분을 반영
def diff_pixel_color(img):
    diff_val = 0
    for y_pos in range(1,val_Y-1):
        for x_pos in range(1,val_X-1):
            #내부 3x3을 탐색한다.
            #bgr이 뒤집혀 있다.
            imp_bgr = [0,0,0]

            for y_small in range (3):
                for x_small in range(3):
                    if (x_small == 1 and y_small == 1):
                        continue
                    else :
                        imp_val = img[y_pos+y_small-1][x_pos+x_small-1]
                        imp_bgr[0] = imp_bgr[0] + imp_val[0]
                        imp_bgr[1] = imp_bgr[1] + imp_val[1]
                        imp_bgr[2] = imp_bgr[2] + imp_val[2]
                        

            
            imp_val = np.subtract(mul_color(img[y_pos][x_pos],8), imp_bgr)
            imp_val2 = val_color(imp_val)

            diff_val =  diff_val + imp_val2
            
    return math.ceil(diff_val/10000)
            

#색의 평균값 구하는 함수==============================================
#단순한 색의 평균값
def avg_pixel_color(img):
    avg_color_per_row = np.average(img,axis=0)
    avg_color = np.average(avg_color_per_row,axis=0)

    new_arr = [0,0,0]
    '''
    avg_color[0] = round(avg_color[0])
    avg_color[1] = round(avg_color[1])
    avg_color[2] = round(avg_color[2])
    '''
    new_arr[0] = round(avg_color[0])
    new_arr[1] = round(avg_color[1])
    new_arr[2] = round(avg_color[2])
    #print(avg_color)
    return new_arr

#색의 분포를 구하는 함수==============================================
#색의 분포를 연산한다. 특정 값에 몰려있는지 구한다.
#count 만큼만 가져올 수 있다. / 0을 넣으면 모두 가져온다.
def distribution_pixel(img,counting):
    dict_dist = {}
    for y_pos in range(1,val_Y-1):
        for x_pos in range(1,val_X-1):
            imp_bgr = [0,0,0]
            for imp_num in range(3):
                imp_bgr[imp_num] = img[y_pos][x_pos][imp_num]
                
            #dict 목적 str만들기
            imp_val_str = (imp_bgr[0] << 16) + (imp_bgr[1] << 8) + (imp_bgr[2])
            
            #distribution 쌓기 작업
            #없으면 새로 추가, 있으면 값1 상승
            if imp_val_str in dict_dist.keys():
                dict_dist[imp_val_str] += 1
            else :
                dict_dist[imp_val_str] = 1

    #dict 완성 후 value 기반으로 sort 한다.
    #내림차순 sort
    list_dist_new = sorted(dict_dist.items(), reverse=True, key=lambda item: item[1])

    if counting != 0 :
        return list_dist_new[0:counting]
    else :
        return list_dist_new
    

def pixel_saturation_applied_average(img):
    #imp_val = 0
    imp_bgr = [0,0,0]

    sat_val = 0
    
    for y_pos in range(0,val_Y):
        for x_pos in range(0,val_X):
            #imp_bgr = [0,0,0]

            val_min = min(img[y_pos][x_pos])
            val_max = max(img[y_pos][x_pos])

            sat_cal = 0
            if val_max == 0:
                sat_cal = 0
            else:
                sat_cal = (val_max-val_min)/val_max

            sat_val += sat_cal
            
            for imp_num in range(3):
                imp_bgr[imp_num] += (sat_cal * img[y_pos][x_pos][imp_num])
                #2 r / 1 g / 0 b  순서이다.
    
    ret_val1 = [imp_bgr[0]/sat_val, imp_bgr[1]/sat_val, imp_bgr[2]/sat_val]
    ret_val2 = sat_val / 100
    return ret_val1, ret_val2

            
#def saturation_average(img):
    
            
    


#=====================================================================
#색의 값을 추출
#위에 64만큼 곱해졌으니 여기서 6만큼 shift
def val_color(arr):
    res = 0
    for i in range(3):
        res = res + ((arr[i]*arr[i]))
        res = res >> 6
    return res

#내부 값 조정, 특정 계수로 곱한다.
def mul_color(arr, mul):
    imp_arr = [0,0,0]
    for i in range(3):
        imp_arr[i] = arr[i]*mul
    return imp_arr

#dist 이미지 표현
def dist_show(img,counting):
    res_list = distribution_pixel(img,counting)
    for val in res_list:
        #RGB 순으로 보여준다.
        print('[ RGB : ' + str(val[0]&255) + ',' + str((val[0]>>8)&255) + ',' + str((val[0]>>16)&255) + ' ] ' + str(val[1]) + '개')


def season_get(int_v):
    if int_v<=2 :
        return 'winter'
    elif (int_v>=3 and int_v <6):
        return 'spring'
    elif (int_v>=6 and int_v <9):
        return 'summer'
    elif (int_v>=9 and int_v <12):
        return 'fall'
    else:
        return 'winter'

#=====================================================================

#배열 생성
file_data = []


#기존 자료 불러오기
with open('data.pickle', 'rb') as f:
      acc = pickle.load(f)

#json 열기
#with open('image_all.json',"r") as json_file:
#    json_data = json.load(json_file)



#앨범 이미지 개별로 불러오기
for i in range(len(img_name_list)):
    #앨범 객체 만들기
    print(str(i) +" 번 진행중 ")
    imp_album = Album()
    imp_album.album = img_name_list[i]

    imp_pub = ''
    imp_check = False

    
    #다른 정보는 acc 에서 찾는다.
    for items in acc :
        if items.album == img_name_list[i]:
            imp_check = True
            imp_album.published_date = items.date
            imp_album.genre.append(items.genre)
            imp_album.title.append(items.title)
            #print(items.title)

    #발견했을 경우에만 적용
    if imp_check == True:

        #월 분리하기
        imp_season = imp_album.published_date[5:7]
        imp_season_int = int(imp_season)
        imp_season_str = season_get(imp_season_int)

        #장르 중복 제거
        imp_arr_genre = set(imp_album.genre)
        imp_arr_genre2 = list(imp_arr_genre)

        #곡 이름 중복 제거
        imp_arr_title = set(imp_album.title)
        imp_arr_title2 = list(imp_arr_title)
        
        #연도 추가
        imp_year = imp_album.published_date[0:4]
        imp_year2 = int(imp_year)

        #분산 연산
        imp_dis = diff_pixel_color(img_list[i])

        #픽셀 평균
        imp_avg = avg_pixel_color(img_list[i])

        #픽셀 채도 곱 평균
        #픽셀 채도 평균
        imp_satM_avg, imp_sat_avg = pixel_saturation_applied_average(img_list[i])
        imp_satM_avg2 = [0,0,0]

        if imp_satM_avg[0] > 0:
            print(imp_satM_avg)
            imp_satM_avg2[0] = round(imp_satM_avg[0])
            imp_satM_avg2[1] = round(imp_satM_avg[1])
            imp_satM_avg2[2] = round(imp_satM_avg[2])
            file_data.append({"album":img_name_list[i],"season":imp_season_str, "genre":imp_arr_genre2, "titles":imp_arr_title2, "year":imp_year2,
                              "pixel_dispersion":imp_dis, "pixel_average":imp_avg, "pixel_saturation_applied_average":imp_satM_avg2, "saturation_average":round(imp_sat_avg,3)})
        

#json 쓰기
with open('image_all.json',"w",encoding='UTF-8-sig') as json_file:
    #json_file.write(json.dump(file_data, ensure_ascii=False))
    json.dump(file_data, json_file, ensure_ascii=False)




'''
data_list = []
data_list_name = []


f = open('data.txt',mode='rt', encoding='utf-8')

while True:
    line = f.readline()

    if not line: break

    data_list.append(line)

    #comma 탐색
    imp_par = line.find(',',7)

    imp_par2 = line.find(',',imp_par + 1)

    imp_title_parsed = line[imp_par+1 : imp_par2]

    #print(imp_title_parsed)

    data_list_name.append(imp_title_parsed)
    
    
f.close


#=====================================================================

#img_list = []
#data_list = []
#data_list_name = []

albums = []

#댄스, 인디음악
#랩/힙합, 인디음악
#발라드, 인디음악

table_genre = ['댄스','랩/힙합','발라드','록/메탈', 'R&B/Soul','일렉트로니카','포크/블루스','성인가요']

for i in range(50):
#for i in range(len(img_list)):
    imp_album = Album()

    #제목 가져오기
    imp_title = file_list[i]

    imp_title = imp_title.replace('.jpg','')
    
    print(imp_title)
    
    imp_img = img_list[i]

    imp_gen = -1

    imp_pos = 0

    imp_str_num = 0

    imp_str_date = ''

    #장르 탐색
    #이미 중복 값이 있으면 99
    for data_num in range(len(data_list)):
        #우선 data_list 내에서 탐색
        if imp_title == data_list_name[data_num]:

            #장르 탐색
            for n in range(len(table_genre)):
                if table_genre[n] in data_list[data_num]:
                    if imp_gen == -1:
                        imp_gen = n
                        imp_pos = data_list[data_num].find(table_genre[n])
                        break

    print(imp_gen)

    imp_str_num = data_list[i].find('20', imp_pos)
    imp_str_date = data_list[i][imp_str_num:imp_str_num+10]
    #print(imp_str_date)
    #print(imp_str_num)
                        
                        
                

    #값 대입
    imp_album.title = file_list[i]
    imp_album.publish_date = imp_str_date
    imp_album.genre = imp_gen
    
    #albums.append(imp_album)


'''    



