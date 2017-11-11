import multiprocessing
import random

class testClass:
    def __init__(self):
        self.special_number = random.random()
        self.assigned = None

    def multNum(self, n):
        return self.special_number * n

def getAllNums(bunchOftesters):
    num = len(bunchOftesters)
