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

json_data_spring = []
json_data_summer = []
json_data_fall = []
json_data_winter = []

genre_table = ["댄스", "랩/힙합", "발라드", "록/메탈", "R&B/Soul", "일렉트로니카", "포크/블루스", "성인가요"]

json_genre_all = []

json_data_genre1 = []
json_data_genre2 = []
json_data_genre3 = []
json_data_genre4 = []
json_data_genre5 = []
json_data_genre6 = []
json_data_genre7 = []
json_data_genre8 = []

json_genre_all.append(json_data_genre1)
json_genre_all.append(json_data_genre2)
json_genre_all.append(json_data_genre3)
json_genre_all.append(json_data_genre4)
json_genre_all.append(json_data_genre5)
json_genre_all.append(json_data_genre6)
json_genre_all.append(json_data_genre7)
json_genre_all.append(json_data_genre8)

with open('image_all.json',"rb") as json_file:
    json_data = json.load(json_file)

for json_values in json_data:
    if json_values["season"] == "spring":
        json_data_spring.append({"season":"spring", "album":json_values["album"], "year":json_values["year"], "pixel_dispersion":json_values["pixel_dispersion"],
                                 "pixel_average":json_values["pixel_average"], "pixel_saturation_applied_average":json_values["pixel_saturation_applied_average"], "saturation_average":json_values["saturation_average"]})
    elif json_values["season"] == "summer":
        json_data_summer.append({"season":"summer", "album":json_values["album"], "year":json_values["year"], "pixel_dispersion":json_values["pixel_dispersion"],
                                 "pixel_average":json_values["pixel_average"], "pixel_saturation_applied_average":json_values["pixel_saturation_applied_average"], "saturation_average":json_values["saturation_average"]})
 
    elif json_values["season"] == "fall":
        json_data_fall.append({"season":"fall", "album":json_values["album"], "year":json_values["year"], "pixel_dispersion":json_values["pixel_dispersion"],
                                 "pixel_average":json_values["pixel_average"], "pixel_saturation_applied_average":json_values["pixel_saturation_applied_average"], "saturation_average":json_values["saturation_average"]})
 
    elif json_values["season"] == "winter":
        json_data_winter.append({"season":"winter", "album":json_values["album"], "year":json_values["year"], "pixel_dispersion":json_values["pixel_dispersion"],
                                 "pixel_average":json_values["pixel_average"], "pixel_saturation_applied_average":json_values["pixel_saturation_applied_average"], "saturation_average":json_values["saturation_average"]})

    for genre_vals in json_values["genre"]:
        for num in range(8):
            if genre_table[num] in genre_vals:
                json_genre_all[num].append({"genre":genre_table[num], "album":json_values["album"], "year":json_values["year"], "pixel_dispersion":json_values["pixel_dispersion"],
                                            "pixel_average":json_values["pixel_average"], "pixel_saturation_applied_average":json_values["pixel_saturation_applied_average"], "saturation_average":json_values["saturation_average"]})


'''
        
with open('season_spring_image.json',"w",encoding='UTF-8-sig') as json_file:
    json.dump(json_data_spring, json_file, ensure_ascii=False)

with open('season_summer_image.json',"w",encoding='UTF-8-sig') as json_file:
    json.dump(json_data_summer, json_file, ensure_ascii=False)

with open('season_fall_image.json',"w",encoding='UTF-8-sig') as json_file:
    json.dump(json_data_fall, json_file, ensure_ascii=False)

with open('season_winter_image.json',"w",encoding='UTF-8-sig') as json_file:
    json.dump(json_data_winter, json_file, ensure_ascii=False)

for nums in range(8):
    
    new_path = 'genre_'+str(nums+1)+'_image.json'
    
    with open(new_path,"w",encoding='UTF-8-sig') as json_file:
        json.dump(json_genre_all[nums], json_file, ensure_ascii=False)
'''

#각 json을 받아서 처리한다.
def show_status(json_values):
    red_val = 0
    green_val = 0
    blue_val = 0

    count = 0
    
    pixel_dispersion = 0
    saturation_average = 0

    red_val2 = 0
    blue_val2 = 0
    green_val2 = 0

    red_val = 0
    blue_val = 0
    green_val = 0
    
    for json_value in json_values:
        count += 1
        pixel_dispersion += json_value["pixel_dispersion"]
        saturation_average += json_value["saturation_average"]

        blue_val += json_value["pixel_saturation_applied_average"][0]
        green_val += json_value["pixel_saturation_applied_average"][1]
        red_val += json_value["pixel_saturation_applied_average"][2]

        blue_val2 += json_value["pixel_average"][0]
        green_val2 += json_value["pixel_average"][1]
        red_val2 += json_value["pixel_average"][2]
        

    print("pixel_dispersion(색상 분산) : "  + str(pixel_dispersion/count))
    print("saturation_average(채도 평균) : " + str(saturation_average/count))

    print("pixel_average(일반 평균 RGB) : R : " + str(red_val2/count) + "/ G : " + str(green_val2/count)  + "/ B : " + str(blue_val2/count))
    
    print("saturation_applied_average(채도 적용 색상 RGB) : R : " + str(red_val/count) + "/ G : " + str(green_val/count)  + "/ B : " + str(blue_val/count))
    print("해당 개수 : " + str(count))
    return 
        
print("봄, 여름, 가을, 겨울 순서")
show_status(json_data_spring)
show_status(json_data_summer)
show_status(json_data_fall)
show_status(json_data_winter)

#print("\n 댄스 랩/힙합 발라드 록/메탈 R&B/Soul 일렉트로니카 포크/블루스 성인가요")
table_num = 0
for genre_val in json_genre_all:
    print('')
    print(genre_table[table_num])
    show_status(genre_val)
    table_num += 1


#-------------------------------------------------

