#naive solution, training on games won with random policy

import os, sys
import pygame
from pygame.locals import *
import math
import random
import numpy as np
import time
# import mxnet
from keras import layers, models, optimizers, callbacks
from keras import backend as K
import glob
import json
import shutil
from PIL import Image
import re
from sklearn.model_selection import train_test_split
import logging
import sys
import pandas as pd
import traceback
import gc
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(stdout_handler)
root_logger.setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
max_record_len = 1000
option_space_size = 4
epsilon = .05



def get_net(x1, x2):
    img_input = layers.Input(shape=(x1.shape[1],), name='img_input')
    input2 = layers.Input(shape=(option_space_size,), name='input2')
    x = layers.concatenate([input2, img_input])
    x = layers.Dense(2048, activation='elu', name='dl1')(x)
    # x = layers.Dropout(.5)(x)
    x = layers.Dense(2048, activation='elu', name='dl2')(x)
    # x = layers.Dropout(.5)(x)
    x = layers.Dense(2048, activation='elu', name='dl3')(x)

    x1 = layers.Dense(1, activation='sigmoid', name='predictions1')(x)

    model = models.Model(inputs = [img_input, input2], outputs = x1, name='dnn')
    model.compile('adam', loss='binary_crossentropy')
    return model



class Agent(pygame.sprite.Sprite):
    def __init__(self, x, y, bounds, range = 20, kill_angle = .02, team = None, a_id = None, alive = True,
                 color = None, pixels_per_square = 8, path = None, g_id = 0, use_model = False, starting_score = 5):
        super(Agent, self).__init__()
        self.x = x
        self.y = y
        self.bounds = bounds
        self.range = range
        self.kill_angle = kill_angle
        self.team = team
        self.a_id = a_id
        self.alive = alive

        self.score= starting_score

        self.width = int(math.sqrt(self.score))
        self.height = int(math.sqrt(self.score))
        self.color = color
        self.pixels_per_square = math.sqrt(self.score)

        self.image = pygame.Surface([self.width, self.height])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.centerx = self.x * self.pixels_per_square
        self.rect.centery = self.y * self.pixels_per_square
        self.alive = True
        self.path = path
        self.g_id = g_id
        self.move_count = 0
        self.logs = []
        self.use_model = use_model
        self.models_loaded = False
        if self.use_model:
            self.load_models()


    def refresh(self, x, y, bounds, g_id = 0):
        self.x = x
        self.y = y
        self.alive = True
        self.bounds = bounds
        self.image = pygame.Surface([self.width, self.height])
        self.image.fill(self.color)
        self.rect = self.image.get_rect()
        self.rect.centerx = self.x * self.pixels_per_square
        self.rect.centery = self.y * self.pixels_per_square
        self.g_id = g_id
        self.move_count = 0
        self.logs = []
        if self.use_model and not self.models_loaded:
            self.load_models()
        # self.load_models()


    #Main function
    def run_turn(self):
        # print('turn for agent', self.a_id, self.x, self.y)
        result = {}
        if self.alive:

            if self.models_loaded:
                if random.random() > epsilon:
                    x = self.get_model_prediction()
                    s_index_np = np.squeeze(x)
                    s_index_np /= s_index_np.sum()
                    s_index_np2 = s_index_np

                    #s_index = float(np.random.choice(np.array([i for i in range(s_index_np2.shape[0])]), p = s_index_np2))
                    # m_index = np.argmax(m_index_np2)
                    s_index = np.argmax(s_index_np2)
                else:
                    s_index = random.choice([i for i in range(option_space_size)])

                # a_m = ((2 * (m_index/option_space_size)) - 1) * math.pi
                a_s = ((2 * (s_index/option_space_size)) - 1) * math.pi

                # x_m = math.cos(a_m)
                # y_m = math.sin(a_m)
                x_s = math.cos(a_s)
                y_s = math.sin(a_s)

            else:
                s_index = random.choice([i for i in range(option_space_size)])
                a_s = ((2 * (s_index/option_space_size)) - 1) * math.pi
                x_s = math.cos(a_s)
                y_s = math.sin(a_s)

            self.logs.append({'s_index':s_index, 'move':self.move_count, 'x':self.rect.centerx, 'y':self.rect.centery})
            # self.move(x_m, y_m)
            result['shot'] = self.shoot(x_s, y_s)
            self.move_count += 1
        return result


    #Action functions
    def move(self, x_m, y_m):
        if self.x + x_m < self.bounds[1] and self.x + x_m >= self.bounds[0]:
            self.x = self.x + x_m
        if self.y + y_m < self.bounds[3] and self.y + y_m >= self.bounds[2]:
            self.y = self.y + y_m




        self.update_center()


    def shoot(self, x_m, y_m):
        a = math.atan2(y_m, x_m)

        # print('pos', (self.x, self.y), (x_m, y_m))
        # print('a', a, math.degrees(a))
        # print('trig', math.cos(a), math.sin(a))
        s = ((self.x, self.y), ((math.cos(a))*self.range + self.x, (math.sin(a))*self.range + self.y))
        s_scaled = ((self.x*self.pixels_per_square, self.y*self.pixels_per_square),
                    (s[1][0]*self.pixels_per_square, s[1][1]*self.pixels_per_square))
        # s = ((self.x, self.y), (x_m, y_m))
        # s_scaled = ((self.x* self.pixels_per_square, self.y* self.pixels_per_square), (x_m* self.pixels_per_square, y_m* self.pixels_per_square))
        shot =  {'origin': (self.x, self.y),
                 'angle': a,
                 'range':self.range,
                 'kill_angle':self.kill_angle,
                 'segment':s,
                 'segment_scaled': s_scaled}
        return shot


    def get_shot_at(self, shot):
        shot_origin = shot['origin']
        angle = math.atan2(self.y - shot_origin[1], self.x - shot_origin[0])
        distance = math.sqrt((self.x - shot_origin[0])**2 + (self.y - shot_origin[1])**2)
        if distance <= shot['range'] and distance > 0 and abs(shot['angle'] - angle) < shot['kill_angle']:
            self.die()


    def die(self):
        self.alive = False


    #Helper functions
    def distance_to_segment(self, seg):
        s1 = seg[0]
        s2 = seg[1]
        t  = abs((((s2[1] - s1[1]) * self.x)) - ((s2[0] - s1[0]) * self.y) + (s2[1]*s1[0]) - (s1[0]*s2[1]))
        b = math.sqrt((s1[1] - s1[0])**2 + (s2[1] - s2[0])**2)
        return t/b


    def update_center(self):
        self.width = int(math.sqrt(self.score))
        self.height = int(math.sqrt(self.score))
        self.color = color
        self.pixels_per_square = math.sqrt(self.score)

        self.image = pygame.Surface([self.width, self.height])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.centerx = self.x * self.pixels_per_square
        self.rect.centery = self.y * self.pixels_per_square

        self.rect.centerx = self.x * self.pixels_per_square
        self.rect.centery = self.y * self.pixels_per_square


    def log_turns(self):
        if self.move_count > 0:
            if not os.path.exists(self.path + '/agent_{0}/'.format(self.a_id)):
                os.makedirs(self.path + '/agent_{0}/'.format(self.a_id))

            with open(self.path + '/agent_{0}/'.format(self.a_id) + 'agent_{0}_{1}.json'.format(self.g_id, self.a_id), 'w') as f:
                # print(self.logs)
                json.dump(self.logs, f)


    def win(self):
        # print(self.logs)
        for i in self.logs:
            i.update({'result':1})
        self.log_turns()


    def lose(self):
        for i in self.logs:
            i.update({'result':0})
        self.log_turns()


    def load_models(self):
        try:
            self.model = models.load_model(self.path + '/agent_{0}/squeezenet'.format(self.a_id))
            with open('/home/td/Documents/rl_tests/swarm_1/dual/a{0}_scaler.plk'.format(self.a_id), 'rb') as f:
                self.scaler = pickle.load(f)
            self.models_loaded = True

        except:
            self.models_loaded = False


    def get_model_prediction(self):
        pass
        img_p = glob.glob(self.path + 'data/last_data.plk')[0]

        with open(img_p, 'rb') as f:
            x1_temp = pickle.load(f)

        x1 = []
        for i in range(option_space_size):
            x1.append(x1_temp)

        x2 = []
        for i in range(option_space_size):
            temp_x2 = [0 for _ in range(option_space_size)]
            temp_x2[i] = 1
            x2.append(temp_x2)

        x1 = np.array(x1)
        x2 = np.array(x2)
        # print(dir(self.scaler))
        x1 = self.scaler.transform(x1)
        x = self.model.predict([x1, x2])

        # img = Image.open(img_p)
        # img = img.resize((128, 128))
        # img = np.array(img)
        #
        # img = img[int(self.rect.centerx) - 64:int(self.rect.centerx) + 64,int(self.rect.centery) - 64:int(self.rect.centery) + 64,:]
        #
        # img = img.astype(np.float32)
        # img /= 255
        # x = self.model.predict(np.array([img]))
        # print(x)
        return x


def train_models(path, a_id, t_id):
    print(path + 'agent_{0}/*.json'.format(a_id))
    result_files = glob.glob(path + 'agent_{0}/*.json'.format(a_id))
    # if len(result_files) > 25000:
    #     result_files = random.sample(result_files, 25000)

    x, y1, y2, results = [], [], [], []
    for i in result_files:
        try:
            with open(i, 'r') as f:
                j = json.load(f)
                if len(j) > max_record_len:
                    continue
                for k in j:
                    # print(re.findall('\d+', i)[-2], k['move'], t_id)

                    img_p = glob.glob(path + 'data/g_img_{0}_{1}_{2}.plk'.format(re.findall('\d+', i)[-2], k['move'], t_id))[0]
                    # img = Image.open(img_p)
                    # img = np.array(img)
                    # img = img[int(k['x']) - 64:int(k['x']) + 64, int(k['y']) - 64:int(k['y']) + 64, :]
                    # img = img.astype(np.float32)
                    # img /= 255

                    with open(img_p, 'rb') as f:
                        img = pickle.load(f)

                    res = k['result']

                    y2_t = int(k['s_index'])
                    y2_t2 = [0 for _ in range(option_space_size)]

                    # y1_t2[y1_t] = 1
                    y2_t2[y2_t] = 1
                    # y1_t2 = np.array(y1_t2)
                    y2_t2 = np.array(y2_t2)

                    x.append(img)
                    # y1.append(y1_t2)
                    y2.append(y2_t2)
                    results.append(res)

        except:
            print(i)
            traceback.print_exc()

    x = np.array(x)

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    with open(path + 'a{0}_scaler.plk'.format(a_id), 'wb') as f:
        pickle.dump(scaler, f)

    y2 = np.array(y2)
    results= np.array(results)
    x_train, x_val, y2_train, y2_val, train_results, val_results = train_test_split(x, y2, results, test_size=.1)

    if x_train.shape[0] < 2000:
        chunk_size = 16
    elif x_train.shape[0] < 4000:
        chunk_size = 32
    elif x_train.shape[0] < 8000:
        chunk_size = 64
    elif x_train.shape[0] < 16000:
        chunk_size = 128
    elif x_train.shape[0] < 32000:
        chunk_size = 256
    elif x_train.shape[0] < 64000:
        chunk_size = 512
    else:
        chunk_size = 1024

    print(x_train.shape, y2_train.shape, train_results.shape)
    net = get_net(x_train, y2_train)
    cb1 = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
    cb2 = callbacks.ModelCheckpoint(path + 'agent_{0}/squeezenet'.format(a_id), monitor='val_loss', verbose=0, save_best_only=True,
                                    save_weights_only=False, mode='auto', period=1)
    net.fit([x_train, y2_train], train_results, validation_data=([x_val, y2_val], val_results), epochs=100, callbacks=[cb1, cb2], batch_size=chunk_size)



class Board():

    def __init__(self, path = None, g_id = 0, agents = 8):

        self.teams = []

        for i in range(agents):
            new_team = [Agent(team = i, a_id = i, bounds=(16, 16, 16, 47), x = random.uniform(16, 47), y = random.uniform(16, 47), color=(int(255 - (i *(agents/255))), int(255 - (i *(agents/1000))), int(-255 + (i *(agents/255)))), g_id = g_id, path=path, range = 30, kill_angle=.2, use_model = random.choice([True, True]))]
            self.teams.append(new_team)

        num_of_food = 100
        self.food = []
        for i in range(num_of_food):
            self.food.append(Agent(team = -1, a_id = -i, bounds=(16, 16, 16, 47), x = random.uniform(16, 47), y = random.uniform(16, 47), color=(128, 128,128), g_id = g_id, path=path, score = random.randrange(1,3)))


    def test_collision(self, a1, a2):
        import math

        if a1.alive and a2.alive:
            if (a1.width/2 + a2.width/2) > math.sqrt((a1.y - a2.y)**2 + (a1.x - a2.x)**2):
                if a1.score > a2.score:
                    a1.score += a2.score
                    a2.alive = False
                elif a2.score > a1.score:
                    a2.score += a1.score
                    a1.alive = False


    def get_board_representation(self):
        data = []

        for t in self.teams:

            for i in t:
                data.append(i.score)
                data.append(i.x)
                data.append(i.y)
                if i.alive:
                    data.append(1)
                else:
                    data.append(0)

            opposing_agents = []
            for i in self.teams:
                if i != t:
                    opposing_agents.extend(i)

            for i in opposing_agents:
                data.append(i.score)
                data.append(i.x)
                data.append(i.y)
                if i.alive:
                    data.append(1)
                else:
                    data.append(0)

                for a in t:
                    self.test_collision(a, i)

            for i in self.food:
                data.append(i.score)
                data.append(i.x)
                data.append(i.y)

                for a in t:
                    self.test_collision(a, i)

        return data


    def refresh(self, path = None, g_id = 0):
        for t in self.teams:
            for count, i in enumerate(t):
                i.refresh( bounds=(16, 47, 16, 47), x = random.randint(16, 47), y = random.randint(16, 47), g_id = g_id)


    def render_agents(self):
        alive_list= []
        for i in self.teams:
            for j in i:
                if j.alive:
                    alive_list.append(j)
        for i in self.food:
            if i.alive:
                alive_list.append(i)

        return alive_list


    def run_turn(self, team):
        shots = []

        team_l = self.teams[team]

        for i in team_l:
             i.run_turn()


        return shots


    def check_if_game_over(self):
        teams_left = 0
        for i in self.teams:
            for j in i:
                if j.alive:
                    teams_left += 1
                    break

        if teams_left > 1:
            return False
        return True



    def get_result(self):
        if self.check_if_game_over():
            for c, i in enumerate(self.teams):
                for j in i:
                    if j.alive:
                        result = [0 for _ in self.teams]
                        result[c] = 1
                        return result



def main():
    path = '/home/td/Documents/rl_tests/swarm_3/agario/'

    if not os.path.exists(path + '/images/'):
        os.makedirs(path + '/images/')
    if not os.path.exists(path + '/data/'):
        os.makedirs(path + '/data/')

    pygame.init()
    screen = pygame.display.set_mode((512, 512))
    pygame.display.set_caption('test')
    pygame.mouse.set_visible(0)
    screen.fill((0, 0, 0))

    st = time.time()
    game_count = 500000
    try:
        result_dicts = pd.read_csv(path + 'res.csv').to_dict(orient='records')
    except:
        result_dicts = []
    game_count_start = 0
    g = game_count_start
    result_dict = {}

    b = Board(path=path, g_id=g)

    while g < game_count:
        screen.fill((0, 0, 0))
        s = b.render_agents()
        for sprite_one in s:
            screen.blit(sprite_one.image, (sprite_one.rect.centerx,sprite_one.rect.centery))
        pygame.display.flip()

        files_to_remove = glob.glob(path + '/images/' + "g_img_{0}_{1}.jpeg".format(g, '*'))
        for i in files_to_remove:
            os.remove(i)
        files_to_remove = glob.glob(path + '/data/' + "g_img_{0}_{1}.jpeg".format(g, '*'))
        for i in files_to_remove:
            os.remove(i)

        count = 0
        while not b.check_if_game_over() and count < max_record_len:
            for turn in [1, 2]:
                gc.collect()
                # pygame.image.save(screen, path + '/images/' + "last_img.jpeg")
                # pygame.image.save(screen, path + '/images/' + "g_img_{0}_{1}_{2}.jpeg".format(g, count, turn))

                data = b.get_board_representation()
                with open(path + '/data/' + "last_data.plk", 'wb') as f:
                    pickle.dump(data, f)
                with open(path + '/data/' + "g_img_{0}_{1}_{2}.plk".format(g, count, turn), 'wb') as f:
                    pickle.dump(data, f)

                time.sleep(.001)
                shots = b.run_turn(turn)
                screen.fill((0, 0, 0))
                for s in shots:
                    # print(s)
                    pygame.draw.line(screen, (0, 255, 0), s[0], s[1])
                s = b.render_agents()
                for sprite_one in s:
                    screen.blit(sprite_one.image, (sprite_one.rect.centerx, sprite_one.rect.centery))
                pygame.display.flip()
                # time.sleep(1)

            count += 1
        # if b.check_if_game_over():
        g += 1
        print(g, time.time() - st)
        b.end_game()
        result = b.get_result()
        result_dict.setdefault(result, 0)
        result_dict[result] += 1
        print('results', result_dict)
        result_dicts.append({'g':g, 'result':result})
        df = pd.DataFrame.from_dict(result_dicts)
        df.to_csv(path + 'res.csv', index=False)
        if g % 5000 == 0 and g > 10000:
            del b
            gc.collect()
            train_models('/home/td/Documents/rl_tests/swarm_1/dual/', 1, 1)
            gc.collect()
            train_models('/home/td/Documents/rl_tests/swarm_1/dual/', 2, 1)
            gc.collect()
            train_models('/home/td/Documents/rl_tests/swarm_1/dual/', 3, 1)
            gc.collect()
            train_models('/home/td/Documents/rl_tests/swarm_1/dual/', 4, 1)
            gc.collect()
            train_models('/home/td/Documents/rl_tests/swarm_1/dual/', 5, 1)
            gc.collect()
            train_models('/home/td/Documents/rl_tests/swarm_1/dual/', 6, 2)
            gc.collect()
            train_models('/home/td/Documents/rl_tests/swarm_1/dual/', 7, 2)
            gc.collect()
            train_models('/home/td/Documents/rl_tests/swarm_1/dual/', 8, 2)
            gc.collect()
            train_models('/home/td/Documents/rl_tests/swarm_1/dual/', 9, 2)
            gc.collect()
            train_models('/home/td/Documents/rl_tests/swarm_1/dual/', 10, 2)
            gc.collect()
            b = Board(path=path, g_id=g)
        b.refresh(path=path, g_id=g)


if __name__ == '__main__':
    # train_models('/home/td/Documents/rl_tests/swarm_1/dual/', 1, 1)
    # train_models('/home/td/Documents/rl_tests/swarm_1/dual/', 2, 2)
    # train_models('/home/td/Documents/rl_tests/swarm_1/dual/', 3, 2)
    # train_models('/home/td/Documents/rl_tests/swarm_1/dual/', 4, 2)
    # train_models('/home/td/Documents/rl_tests/swarm_1/dual/', 5, 2)
    # train_models('/home/td/Documents/rl_tests/swarm_1/dual/', 6, 2)
    # train_models('/home/td/Documents/rl_tests/swarm_1/dual/', 7, 2)
    # train_models('/home/td/Documents/rl_tests/swarm_1/dual/', 8, 2)
    # train_models('/home/td/Documents/rl_tests/swarm_1/dual/', 9, 2)
    # train_models('/home/td/Documents/rl_tests/swarm_1/dual/', 10, 2)
    main()
