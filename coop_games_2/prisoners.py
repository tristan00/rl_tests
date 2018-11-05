import lightgbm
import random


class DBot():
    def __init__(self, n):
        self.b_id = n
        self.game_history = []




class Game():
    def __init__(self, b1, b2, rounds = 100):

        round_history = []
        

        for i in rounds:
            b1_data = b1.get_data()
            b2_data = b2.get_data()


bots = [DBot(i) for i in range(10)]


