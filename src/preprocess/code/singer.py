
class Singer:
    def __init__(self):
        self.name = ''
        self.sex = ''
        self.group = ''
        self.fan = 0

    def getRow(self):
        return [self.name, self.sex, self.group, self.fan]