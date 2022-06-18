# Analysis of Chart using Music Data
- 최신기술 프로젝트 3조 팀 프로젝트입니다.

## Project
음원 데이터를 활용한 차트 분석
데이터에 대한 주제별 분석과 시각화, 이를 통한 insight 도출
![그림0](https://user-images.githubusercontent.com/56705742/121757494-b342eb80-cb58-11eb-9531-8045c02f64c7.png)


### Motivation & Goal
![image](https://user-images.githubusercontent.com/56705742/121767281-3cc2df80-cb92-11eb-963c-a307bc88a472.png)
음악은 현대인의 여기생활에서 가장 자주 활용됩니다. 인간에게 큰 영향을 미치는 만큼 이를 주제로 분석을 진행하기로 결정했습니다. 무수히 많은 노래에서 대중들에게 사랑 받는 top100 차트의 음악을 대상으로 그들의  흥행 요인을 분석하기로 결정했습니다.

관련 데이터를 수집하고 해당 데이터에 대해 5가지의 주제를 분석하고 유의미한 insight를 도출하는 것을 목표로 진행했습니다.
1. 장르별, 연도별 인기있는 아티스트 형태
2. 연도별, 계절별 인기있는 장르의 분석
3. 장르, 연도에 따라 가사에서 가장 많이 사용된 키워드
4. 가사의 키워드 유사도에 따른 노래 추천
5. 장르에 따른 앨범 채색 분석

### Installing
~~~
$ git clone https://github.com/philjjoon/2021-01-GROUP-03'
~~~

## Data Acquisition & Preprocessing
데이터 분석을 위한 데이터 습득과 전처리 과정

### Data Acquisition
~~~
$ cd src/preprocess/code
~~~
1. 데이터 클래스 구조 생성 Song class, Singer class 
~~~
class Song:
    def __init__(self):
        self.year = 0
        self.month = 0
        self.rank = 0
        self.title = ''
        self.singer = ''
        self.album = ''
        self.genre = ''
        self.date = ''
        self.likes = 0
        self.lyrics = ''
        self.keywords = []

    def setImage(self, url):
        res = requests.get(url)
        self.img = Image.open(BytesIO(res.content))
~~~

~~~
class Singer:
    def __init__(self):
        self.name = ''
        self.sex = ''
        self.group = ''
        self.fan = 0

~~~

2. https://www.melon.com/chart/search/index.htm 에 대해 selenium crawling 진행, 반환 리스트에 저장
![image](https://user-images.githubusercontent.com/56705742/121764545-b1d8e980-cb7f-11eb-9c6c-5529756f4e3d.png)

수집한 데이터에 대해 set를 사용해서 중복을 제거
~~~
data = []
for filename in range(1112, 2122, 202):
    with open(str(filename)+'.pickle', 'rb') as f:
        tmp = pickle.load(f)
    data.extend(tmp)

singers = set()
for d in data:
    singers.add(d.singer)

with open('singer_name.pickle', 'wb') as f:
    pickle.dump(list(singers), f)
~~~

### Data Preprocessing
1. 코모란을 통한 핵심 키워드 추출, 영어 단어 대상으로 nltk를 통한 추가 태깅 작업
~~~
def komoran_tokenize(sent):
    words = sent.split()
    for i in range(len(words)):
        if words[i].endswith('/SL') and len(words[i]) > 4:
            words[i] = words[i][:-3]
            words[i] = '/'.join(nltk.pos_tag(nltk.word_tokenize(words[i]))[0])
            if words[i].endswith('/NN'):
                words[i] += 'P'
    words = [w for w in words if '/NNP' in w or '/NNG' in w  or '/FW' in w or '/JJ' in w]

    return words
~~~

2. pagerank 알고리즘 적용, 주요 키워드 선정
사이킷런의 희소 행렬을 사용하여 단어 사전을 만든다.
![image](https://user-images.githubusercontent.com/56705742/121764579-ffeded00-cb7f-11eb-98ad-c9d970401fff.png)

![image](https://user-images.githubusercontent.com/56705742/121757996-914a6880-cb5a-11eb-831a-847cb8e5a0f8.png)

3. 피클 파일 생성, csv파일로 변환. Song.csv Singer.csv를 생성
~~~
f = open('data.csv', 'w', newline='', encoding='UTF-8')
wr = csv.writer(f)
komoran = Komoran('STABLE')

for i in range(len(data)):
    # 제목 정제
    idx = data[i].title.find('(')
    if idx != -1:
        data[i].title = data[i].title[:idx]
    # 가사 정제
    if data[i].lyrics != '' and data[i].title != '거꾸로 걷는다':
        texts = data[i].lyrics.split('\n')
        sents = []
        for text in texts:
            tokened_text = komoran.get_plain_text(text)
            if tokened_text != '':
                sents.append(tokened_text)
        keyword_extractor = KeywordSummarizer(
            tokenize = komoran_tokenize,
            window = -1,
            verbose = False
        )
        if len(sents) != 0:
            keywords = keyword_extractor.summarize(sents, topk=5)
            data[i].keywords = list(map(lambda x : x[0][:x[0].find('/')], keywords))

    wr.writerow(data[i].getRow())
    data[i].saveImg()

f.close()
~~~
![image](https://user-images.githubusercontent.com/56705742/121758013-9f988480-cb5a-11eb-91d0-4abed597d09b.png)

4. 본인의 하둡 HDFS에 적재
![image](https://user-images.githubusercontent.com/56705742/121758018-a45d3880-cb5a-11eb-9494-09fd3cb1e702.png)

## Data Analysis & Visualization
생성된 데이터에 대해 5가지의 주제의 분석 진행
1. 장르별, 연도별 인기있는 아티스트 형태
2. 연도별, 계절별 인기있는 장르의 분석
3. 장르, 연도에 따라 가사에서 가장 많이 사용된 키워드
4. 가사의 키워드 유사도에 따른 노래 추천
5. 장르에 따른 앨범 채색 분석

### 1. 장르별, 연도별 인기있는 아티스트 형태

#### Data analysis
~~~
$ cd src/analysis/code
~~~

1. 연도 계절별 차트 나누기
~~~
years = [2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]

for year in years:
    df_year = joined_df.filter(joined_df.year == year)
~~~

~~~
seasons = [[3,4,5,'spring'],[6,7,8,'summer'],[9,10,11,'fall'],[12,1,2,'winter']]

for s in seasons:
    df_season = joined_df.filter(joined_df.month.isin(s))
~~~


2. 곡 별로 인기의 수준 계산
변수 x는 차트 순위이고 1위에 100점, 100위에 1점 순으로 할당
변수 d는 파생변수로 해당 순위에 머물렀던 개월 수에 해당
![image](https://user-images.githubusercontent.com/56705742/121758128-1a619f80-cb5b-11eb-8169-b68d725e9771.png)

3. 아티스트별로 인기의 수준 계산
~~~
    artist_score_df = score_by_rank_df.groupBy('artist','gender','type').agg(F.sum('score'))
    artist_score_df = artist_score_df.withColumnRenamed('sum(score)', 'score')
    artist_score_df = artist_score_df.withColumn('year',lit(year))
    artist_score_df = artist_score_df.sort('score',ascending=False)
~~~

4. 아티스트를 성별과 솔로/그룹 여부에 따라 평균 점수 연산
~~~
    gender_type_result = artist_score_df.groupBy('gender','type').agg(F.mean('score'))
    gender_type_result = gender_type_result.withColumn('year',lit(year))
    gender_type_result = gender_type_result.sort('avg(score)', ascending=False)
~~~
#### Visualization
위에서 구한 json파일들에 시각화 진행
* 연도별 인기 있던 장르 분석 결과
13-16년도 사이의 혼성 아티스트의 붐
![image](https://user-images.githubusercontent.com/56705742/121758212-77f5ec00-cb5b-11eb-814f-74643bab06a7.png)
최근 3년간의 남성 그룹의 인기
![image](https://user-images.githubusercontent.com/56705742/121758238-922fca00-cb5b-11eb-93eb-2e50fb0fc8e6.png)

* 연도별 점수에 따른 상위권 아티스트의 분석
![image](https://user-images.githubusercontent.com/56705742/121758268-affd2f00-cb5b-11eb-92ca-f2d19ac11fe8.png)

### 2. 연도별, 계절별 인기있는 장르의 분석
#### Data analysis
1. 계절 별로 차트를 나누기

~~~
genre_score_df = season_genre_df.withColumn('score', (101 - F.col('rank'))*F.col('cnt'))
genre_score_df = genre_score_df.drop('rank').drop('cnt').groupBy('genre','year','title','artist').agg(F.sum('score'))
genre_score_df = genre_score_df.withColumnRenamed('sum(score)', 'score')
~~~

2. 곡 별로 인기 수준을 계산
~~~
genre_rank = genre_score_df.groupBy('genre','year').agg(F.avg('score'))
genre_rank = genre_rank.withColumnRenamed('avg(score)', 'score')
genre_rank = genre_rank.sort('year','score',ascending=False)
genre_rank = genre_rank.withColumn('season',lit(s[3])
~~~

#### Visualization
* 인기 장르 동향
![image](https://user-images.githubusercontent.com/56705742/121760177-59e0b980-cb64-11eb-9489-8179e70b6114.png)

* 2017년일 경우 장르별 그룹화 했을 시 고득점
![image](https://user-images.githubusercontent.com/56705742/121760173-56e5c900-cb64-11eb-8e75-4ed8664ecd37.png)


### 3. 장르, 연도에 따라 가사에서 가장 많이 사용된 키워드

#### Data analysis
1. HDFS에서 csv 파일 읽어오기, 장르, 연도, 시대, 계절 별로 키워드 추출
~~~
def select_keyword(keywords,i):
    keywords = keywords.replace("[","").replace("]","").replace("'","").replace(" """).split(",")
    if len(keywords)-1 < i:
        return ""
    else:
        return keywords[i]
~~~

2. 특성별로 키워드 개수를 카운팅, json으로 저장
위의 데이터를 키워드, 장르 기반으로 재구성
키워드 별로 새로운 행을 생성한다. 

장르와 동일하게 연도 기반으로 재구성
![image](https://user-images.githubusercontent.com/56705742/121766489-03d43c00-cb8d-11eb-9409-6198ceb88d66.png)
set season 함수를 통해 새로운 month 열 생성
![image](https://user-images.githubusercontent.com/56705742/121760317-0cb11780-cb65-11eb-86d6-1790693f02b3.png)
3. 추출 값에 대한 word count 진행
![image](https://user-images.githubusercontent.com/56705742/121760351-497d0e80-cb65-11eb-923e-c7621d939ada.png)

![image](https://user-images.githubusercontent.com/56705742/121760350-46821e00-cb65-11eb-9ac0-02e43f12ae43.png)


#### Visualization
* 시대별 키워드에 대한 시각화
영어의 사용 빈도 증가의 경향
![image](https://user-images.githubusercontent.com/56705742/121760366-5ac61b00-cb65-11eb-8386-1dc5abbe1ed7.png)

* 장르별 키워드에 대한 시각화
장르별로 자주 등장하는 단어의 비중이 크게 다르다.
![image](https://user-images.githubusercontent.com/56705742/121760383-6fa2ae80-cb65-11eb-8d00-ddeb860a67e1.png)
![image](https://user-images.githubusercontent.com/56705742/121760401-85b06f00-cb65-11eb-9999-1bc666739f95.png)
계절에 따라 키워드의 빈도수가 달라짐을 확인 가능
![image](https://user-images.githubusercontent.com/56705742/121760414-9365f480-cb65-11eb-8a9d-16b06dbfcd45.png)

### 4. 가사의 키워드 유사도에 따른 노래 추천


#### Data analysis
1. 곡에 포함된 가사 기반 TF-IDF 모델 설계
TF-IDF 모델은 단순 빈도수 만이 아닌 단어의 중요도를 판정해 빈도의 스케일을 조정한다.
TF(증가 빈도)는 특정 단어의 빈도수를 계산한다. 상대적인 출현 빈도를 계산한다.
IDF는(역문서 빈도)는 한 단어가 문서 전체에서 공통적으로 나타나는지 나태낸다. 이 값이 높으면 공통적으로 나타나는 빈도가 낮다고 할 수 있다.

![image](https://user-images.githubusercontent.com/56705742/121760604-96151980-cb66-11eb-9ab3-0e82890fa514.png)

둘의 곱은  해당 문서 내에서 대상 단어의 출현 빈도가 높으나 그 단어를 포함한 문서는 적을 수록 증가하게 된다. 무의미하게 증가하는 단어는 배제하면서 빈도 수를 계산할 수 있다.

2. 작업 대상 가사를 워드 벡터로 만든다
이를 위해 pyspark feature의 토크나이저를 사용
~~~
tokenizer = Tokenizer(inputCol="lyrics", outputCol="words\")
wordsData = tokenizer.transform(chart_df)
~~~

3. 워드 벡터 값에 대해 TF-IDF 적용
~~~
hashingTF = HashingTF(inputCol="words", outputCol="tf")
tfData = hashingTF.transform(wordsData)

idf = IDF(inputCol="tf", outputCol="feature")
idfModel = idf.fit(tfData)
data = idfModel.transform(tfData)

data.toPandas()
~~~

4. 이 TF-IDF 벡터를 기준으로 cosine 유사도 계산
~~~
    def dot(v1,v2):
        return sum([x * y for x,y in zip(v1,v2)])

    def L2(v)
        return math.sqrt(sum([x ** 2 for x in v]))

    @udf
    def sim_cos(v1,v2):
        try:
            return float(dot(v1,v2))/float(L2(v1)*L2(v2))
        except:
            return 0
~~~

~~~
result = data.alias("i").join(data.alias("j"), F.col("i.ID") < F.col("j.ID"))
    .select(\n",
        F.col("i.ID").alias("i")
        F.col("j.ID").alias("j")
        sim_cos("i.feature", "j.feature").alias("sim_cosine"))
    .sort("i", "j")"
~~~
각 row마다 가사를 뽑아 counting을 진행한다.
![image](https://user-images.githubusercontent.com/56705742/121760817-aaa5e180-cb67-11eb-8497-7c6a892d087a.png)

5. 워드카운트를 위한 json화
~~~
import json

    def rec_to_actions(df, INDEX, TYPE, cnt):
        idx = cnt
        print(idx)
        lst = []
        for record in df.to_dict(orient=\"records\"):
            lst.append(('{ \"index\" : { \"_index\" : \"%s\", \"_type\" : \"%s\", \"_id\" : %d }}'% (INDEX, TYPE,idx)))
            lst.append(json.dumps(record, default=str, ensure_ascii=False))
            idx += 1
        return lst
~~~

~~~
cnt = 0

result = rec_to_actions(song_count.toPandas(),"lyrics_similarity", "l_s", cnt)

dest_file = 'elk_keyword/lyrics_similarity.json
output_file = open(dest_file, 'w', encoding='utf-8')

for l in result:
    print(eval(l))
    json.dump(eval(l), output_file, ensure_ascii=False)
    output_file.write("\n")
~~~
#### Visualization
* '매력있어'라는 곡을 넣고 확인
![image](https://user-images.githubusercontent.com/56705742/121760842-c610ec80-cb67-11eb-8056-967d9d1290ea.png)

### 5. 장르에 따른 앨범 채색 분석
~~~
$ cd src/preprocess/code
~~~
#### Data analysis
1. 이미지 압축 변환
preprocess
~~~
val_Y = 100
val_X = 100

path_dir = './scr/preprocess/results/images'

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
~~~

2. 이미지 폴더와 차트 데이터를 통한 album data 생성
하부에서 설명할 함수와 함께 작업 진행
~~~
for i in range(len(img_name_list)):
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
~~~

3. dispersion, saturation, saturation applied RGB value 산출
탐색 대상과 주위 픽셀 값들의 평균 차이를 구해 분산의 합을 연산
해당 값이 높을수록 이미지 내 고주파 값이 높음을 의미한다.
~~~
def diff_pixel_color(img):
    diff_val = 0
    for y_pos in range(1,val_Y-1):
        for x_pos in range(1,val_X-1):
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
~~~
![image](https://user-images.githubusercontent.com/56705742/121766276-bd321200-cb8b-11eb-9dd7-bc94b33d57a6.png)
채도는 개별 pixel의 값을 통해 연산할 수 있다.

RGB값에 채도 값을 곱함으로서 무채색 픽셀의 영향력을 줄인다.
~~~
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
~~~

4. 계절별, 장르별 json 파일로 재구성
~~~
genre_table = ["댄스", "랩/힙합", "발라드", "록/메탈", "R&B/Soul", "일렉트로니카", "포크/블루스", "성인가요"]

for genre_vals in json_values["genre"]:
        for num in range(8):
            if genre_table[num] in genre_vals:
                json_genre_all[num].append({"genre":genre_table[num], "album":json_values["album"], "year":json_values["year"], "pixel_dispersion":json_values["pixel_dispersion"],
                                            "pixel_average":json_values["pixel_average"], "pixel_saturation_applied_average":json_values["pixel_saturation_applied_average"], "saturation_average":json_values["saturation_average"]})
~~~
#### Visualization
* 장르별 분산, 채도
랩 힙합 부분에서 높은 채도와 분산을 확인 가능
반대로 발라드에서는 채도와 분산이 상당히 낮아진다.
![image](https://user-images.githubusercontent.com/56705742/121766318-e488df00-cb8b-11eb-8b80-0bf43b5ecdd9.png)

* 계절별 평균 RGB 값의 변화
계절별로는 그 수치가 계절의 진행에 따라 일괄적으로 감소하는 경향이 주로 나타난다.
![image](https://user-images.githubusercontent.com/56705742/121766344-0e420600-cb8c-11eb-8788-bc3496c3bd67.png)
