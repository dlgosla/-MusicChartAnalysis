import requests, re
from io import BytesIO
from PIL import Image
import numpy as np

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

    def getRow(self):
        self.title.strip()
        self.album = re.sub(r"[?'/\"*<>:]", "", self.album)
        try:
            return [self.year, self.month, self.rank, self.title, self.singer, self.album, self.genre, self.date, self.likes, self.keywords]
        except:
            return [self.year, self.month, self.rank, self.title, self.singer, self.album, self.genre, self.date, self.likes, []]

    def saveImg(self):
        self.img.save('./images/' + self.album + '.jpg')