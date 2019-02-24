import os, sys
import pygame
import random
import math
import copy
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow
from keras import layers, models, callbacks
import os
import glob
import json
import pandas as pd

import traceback


screen_size = 512
epsilon = .1
allowed_moves = [0, 1, 2, 3]
agent_step_size = 2.5
path = r'C:\Users\trist\Documents\agar_data/'
color_number = random.randint(1000000, 2000000)
max_samples_per_game = 100


def get_net(x_a, x_b):
    x_in = layers.Input(shape=(x_a.shape[1],), name='img_input')
    x1 = layers.Dense(256, activation='relu', name='dl1')(x_in)
    x2 = layers.Dense(256, activation='relu', name='dl2')(x1)
    x3 = layers.Dense(256, activation='relu', name='dl3')(x2)
    x_out = layers.Dense(x_b.shape[1], activation='sigmoid', name='predictions1')(x3)
    model = models.Model(inputs = x_in, outputs = x_out, name='dnn')
    model.compile('adam', loss='binary_crossentropy')
    return model


class Agent(pygame.sprite.Sprite):
    def __init__(self, x, y, color, a_id, score = 100):
        super(Agent, self).__init__()
        self.x = x
        self.y = y
        self.screen = None
        self.a_id = a_id
        self.color = color
        self.score = score
        self.alive = 1

        self.b1 = 0
        self.b2 = screen_size
        self.move_list = []

        self.model = None
        self.model_trained = False
        self.starting_score = score

    def redraw(self):
        pygame.draw.circle(self.screen, self.color, (int(self.x), int(self.y)), int(math.sqrt(self.score)))


    def refresh(self, x, y):
        self.alive = 1
        self.x = x
        self.y = y
        self.screen = None
        self.score = 100
        self.score = self.starting_score


    def train_model(self):
        files = glob.glob(path + "/data_{0}/*.pkl".format(self.a_id))

        x = []
        y = []

        for i in files:
            with open(i, 'rb') as f:
                try:
                    new_data = pickle.load(f)

                    new_x = new_data['x']
                    new_y = new_data['y']

                    if len(new_x) > max_samples_per_game:
                        x1, x2, y1, y2 = train_test_split(new_x, new_y, test_size=max_samples_per_game)

                        x.extend(x2)
                        y.extend(y2)
                except:
                    print(i)
                    traceback.print_exc()

        x = np.array(x)
        y = np.array(y)

        self.model = get_net(x, y)
        self.scaler = StandardScaler()
        x = self.scaler.fit_transform(x)
        x1, x2, y1, y2 = train_test_split(x, y, test_size=.1)
        cb1 = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
        cb2 = callbacks.ModelCheckpoint(path + 'data_{0}/squeezenet'.format(self.a_id), monitor='val_loss', verbose=0,
                                        save_best_only=True,
                                        save_weights_only=False, mode='auto', period=1)
        self.model.fit(x1, y1, validation_data=(x2, y2), epochs=5,
                callbacks=[cb1, cb2], batch_size=128, verbose = 0)

        self.model_trained = True


    def run_move(self, data):

        if random.random() < epsilon or not self.model_trained:
            move = self.random_move()
        else:
            move = self.model_move(data)
        self.execute_move(move)


    def model_move(self, data):

        x_2 = [[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]]

        all_x = [i + data for i in x_2]
        all_x = np.array([self.scaler.transform(np.array([i])) for i in all_x])
        out = self.model.predict(np.squeeze(all_x))
        return np.argmax(out[:, self.a_id])



    def random_move(self):
        move = random.choice(allowed_moves)
        return move



    def execute_move(self, move):
        move_array = [0 for _ in allowed_moves]
        move_array[move] = 1
        self.move_list.append(move_array)
        if move == 0:
            self.x = self.x + agent_step_size
            self.y = self.y + 0
        if move == 1:
            self.x = self.x + 0
            self.y = self.y + agent_step_size
        if move == 2:
            self.x = self.x - agent_step_size
            self.y = self.y + 0
        if move == 3:
            self.x = self.x + 0
            self.y = self.y - agent_step_size

        if self.x < self.b1:
            self.x = self.b1
        if self.x > self.b2:
            self.x = self.b2
        if self.y < self.b1:
            self.y = self.b1
        if self.y > self.b2:
            self.y = self.b2




    def record_results(self, g_x, y, path, g_id):
        data = [(a + b, c) for a, b, c in zip(self.move_list, g_x, y)]
        x = [i[0] for i in data]
        y = [i[1] for i in data]

        if not os.path.exists(path + "/data_{0}/".format(self.a_id)):
            os.makedirs(path + "/data_{0}/".format(self.a_id))
        with open(path + "/data_{0}/{1}.pkl".format(self.a_id, g_id), 'wb') as f:
            pickle.dump({'x': x, 'y': y}, f)


class Game():

    def __init__(self, agent_count = 4, food_count = 100, max_rounds = 1000, g_id = 0, min_training_games = 1000):
        self.agents = [Agent(a_id = i, x = random.uniform(16, screen_size - 16), y = random.uniform(16, screen_size - 16), color = (int((color_number*(i + 1))%255), int((color_number*(i + 1)*7)%255), int((color_number*(i + 1)*11)%255))) for i in range(agent_count)]
        self.food = [Agent(a_id = i, x = random.uniform(16, screen_size - 16), y = random.uniform(16, screen_size - 16), color = (255,255,255), score = random.randint(5, 50)) for i in range(food_count)]
        self.max_rounds = max_rounds
        self.food_count = food_count
        self.agent_count = agent_count
        self.g_id =g_id
        self.min_training_games = min_training_games

        if g_id >= min_training_games:
            for i in self.agents:
                import traceback
                try:
                    i.train_model()
                except:
                    traceback.print_exc()


    def set_screen(self, screen):
        self.screen = screen
        for i in self.agents:
            i.screen = screen
        for i in self.food:
            i.screen = screen


    def refresh_game(self, g_id, retrain = False):
        self.g_id = g_id

        for i in self.agents:
            i.refresh(x = random.uniform(16, screen_size - 16), y = random.uniform(16, screen_size - 16))
        for i in self.food:
            i.refresh(x = random.uniform(16, screen_size - 16), y = random.uniform(16, screen_size - 16))

        #
        # self.agents = [Agent(a_id = i, x = random.uniform(16, screen_size - 16), y = random.uniform(16, screen_size - 16), color = (int((color_number*(i + 1))%255), int((color_number*(i + 1)*7)%255), int((color_number*(i + 1)*11)%255))) for i in range(self.agent_count)]
        # self.food = [Agent(a_id = i, x = random.uniform(16, screen_size - 16), y = random.uniform(16, screen_size - 16), color = (255,255,255), score = random.randint(5, 50)) for i in range(self.food_count)]
        #
        if g_id >= self.min_training_games and retrain:
            for i in self.agents:
                import traceback
                try:
                    i.train_model()
                except:
                    traceback.print_exc()


    def run_game(self):
        data_list = []

        for i in range(self.max_rounds):
            new_data = self.get_board_representation()
            data_list.append(new_data)

            for j in self.agents:
                if j.alive:
                    j.run_move(new_data)

            for j in self.food:
                j.redraw()

            for j in self.agents:
                for k in self.agents + self.food:
                    self.test_collision(j, k)

            self.screen.fill((0, 0, 0))
            for j in self.agents + self.food:
                if j.alive:
                    j.redraw()
            pygame.display.flip()

        score_id_list = [(i.a_id, i.score, i.alive) for i in self.agents]
        score_id_list.sort(key = lambda x : x[1])

        game_stats = [{'a_id':i[0], 'score':i[1], 'alive':i[2], 'g_id':self.g_id} for i in score_id_list]
        winner= game_stats[-1]['a_id']
        winner_array = [0 for _ in self.agents]
        winner_array[winner] = 1
        y = [winner_array for _ in range(self.max_rounds)]

        for j in self.agents:
            j.record_results(data_list, y, path, self.g_id)


        print(self.g_id, game_stats)
        return pd.DataFrame.from_dict(game_stats)


    def get_board_representation(self):
        data = []
        for i in self.agents:
            data.append(i.score)
            data.append(i.alive)

            for j in self.agents:
                if i != j:
                    data.append(i.x - j.x)
                    data.append(i.y - j.y)
                    data.append(math.atan2(i.y - j.y, i.x - j.x))

            for j in self.food:
                data.append(i.x - j.x)
                data.append(i.y - j.y)
                data.append(math.atan2(i.y - j.y, i.x - j.x))

        for i in self.food:
            data.append(i.score)
            data.append(i.alive)

        return data


    def test_collision(self, a1, a2):
        if a1.alive and a2.alive and a1 != a2:
            if (math.sqrt(a1.score)/2 + math.sqrt(a2.score)/2) > math.sqrt((a1.y - a2.y)**2 + (a1.x - a2.x)**2):
                if a1.score > a2.score:
                    a1.score += a2.score
                    a2.alive = 0
                elif a2.score > a1.score:
                    a2.score += a1.score
                    a1.alive = 0



def run_game():


    game_stats = []
    pygame.init()

    g = Game(g_id=0)

    for i in range(10000):
        screen = pygame.display.set_mode((screen_size, screen_size))
        pygame.event.get()

        if i%100 == 0:
            g.refresh_game(i, retrain=True)
        else:
            g.refresh_game(i, retrain=False)
        g.set_screen(screen)

        pygame.display.set_caption('test')
        pygame.mouse.set_visible(0)
        screen.fill((0, 0, 0))
        game_stats.append(g.run_game())
        del screen

        if i % 200 ==0 and i > 0:
            full_stats = pd.concat(game_stats)
            full_stats.to_csv('game_stats.csv')



run_game()