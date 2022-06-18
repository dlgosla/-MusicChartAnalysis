import pickle
from song import *

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