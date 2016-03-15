class BreakPoint(object):
    def __init__(self, avg, name1, name2):
        self.avg = avg
        self.name1 = name1
        self.name2 = name2

    def __str__(self):
        return "AVG: {} Name1: {} Name2: {}".format(self.avg, self.name1, self.name2)