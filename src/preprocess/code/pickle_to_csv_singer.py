import csv, pickle
from singer import *


data = []
with open('./data/singer.pickle', 'rb') as f:
    data = pickle.load(f)

f = open('./data/singer.csv', 'w', newline='', encoding='UTF-8')
wr = csv.writer(f)

for singer in data:
    wr.writerow(singer.getRow())

f.close()