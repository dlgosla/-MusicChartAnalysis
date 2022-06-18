import re, csv, pickle, nltk
from song import *
from PyKomoran import *
from textrank import KeywordSummarizer

#nltk.download('averaged_perceptron_tagger')

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

data = []
for filename in range(1112, 2122, 202):
    with open(str(filename)+'.pickle', 'rb') as f:
        tmp = pickle.load(f)
    data.extend(tmp)

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