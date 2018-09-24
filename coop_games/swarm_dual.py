#naive solution, training on games won with random policy

import os, sys
import pygame
from pygame.locals import *
import math
import random
import numpy as np
import time
import mxnet
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


root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(stdout_handler)
root_logger.setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
max_record_len = 200
option_space_size = 100



def get_net():
    img_input = layers.Input(shape=(128, 128, 3))
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)


    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(4096, activation='relu', name='dl1')(x)
    x = layers.Dense(4096, activation='relu', name='dl2')(x)

    x1 = layers.Dense(option_space_size, activation='softmax', name='predictions1')(x)
    x2 = layers.Dense(option_space_size, activation='softmax', name='predictions2')(x)

    model = models.Model(inputs = img_input, outputs = [x1, x2], name='vgg16')
    model.compile('adam', loss='categorical_crossentropy')
    return model


class Agent(pygame.sprite.Sprite):
    def __init__(self, x, y, bounds, range = 20, kill_angle = .02, team = None, a_id = None, alive = True,
                 color = None, pixels_per_square = 8, path = None, g_id = 0, use_model = False):
        super(Agent, self).__init__()
        self.x = x
        self.y = y
        self.bounds = bounds
        self.range = range
        self.kill_angle = kill_angle
        self.team = team
        self.a_id = a_id
        self.alive = False
        self.width = pixels_per_square
        self.height = pixels_per_square
        self.color = color
        self.image = pygame.Surface([self.width, self.height])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.centerx = self.x * pixels_per_square
        self.rect.centery = self.y * pixels_per_square
        self.pixels_per_square = pixels_per_square
        self.alive = alive
        self.path = path
        self.g_id = g_id
        self.move_count = 0
        self.logs = []
        self.use_model = use_model
        if self.use_model:
            self.load_models()


    #Main function
    def run_turn(self):
        # print('turn for agent', self.a_id, self.x, self.y)

        result = {}

        if self.alive:
            #TODO: replace random with 3 networks

            if  self.use_model:
                x = self.get_model_prediction()
                # print(x)
                # x = np.squeeze(x)
                # x_m_np = x[0:option_space_size]
                # x_s_np = x[option_space_size:]
                m_index_np = x[0]
                m_index_np /= m_index_np.sum()
                m_index_np2 = m_index_np[0]
                s_index_np = x[1]
                s_index_np /= s_index_np.sum()
                s_index_np2 = s_index_np[0]

                m_index = float(np.random.choice(np.array([i for i in range(m_index_np2.shape[0])]), p = m_index_np2))
                s_index = float(np.random.choice(np.array([i for i in range(s_index_np2.shape[0])]), p = s_index_np2))

                a_m = ((2 * (m_index/option_space_size)) - 1) * math.pi
                a_s = ((2 * (s_index/option_space_size)) - 1) * math.pi

                x_m = math.cos(a_m)
                y_m = math.sin(a_m)
                x_s = math.cos(a_s)
                y_s = math.sin(a_s)

            else:

                m_index = random.choice([i for i in range(option_space_size)])
                s_index = random.choice([i for i in range(option_space_size)])

                a_m = ((2 * (m_index/option_space_size)) - 1) * math.pi
                a_s = ((2 * (s_index/option_space_size)) - 1) * math.pi

                x_m = math.cos(a_m)
                y_m = math.sin(a_m)
                x_s = math.cos(a_s)
                y_s = math.sin(a_s)


            self.logs.append({'m_index':m_index, 's_index':s_index, 'move':self.move_count, 'x':self.rect.centerx, 'y':self.rect.centery})
            self.move(x_m, y_m)
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
        self.rect.centerx = self.x * self.pixels_per_square
        self.rect.centery = self.y * self.pixels_per_square


    def log_turns(self):
        if self.move_count < max_record_len:
            if not os.path.exists(self.path + '/agent_{0}/'.format(self.a_id)):
                os.makedirs(self.path + '/agent_{0}/'.format(self.a_id))

            with open(self.path + '/agent_{0}/'.format(self.a_id) + 'agent_{0}_{1}.json'.format(self.g_id, self.a_id), 'w') as f:
                # print(self.logs)
                json.dump(self.logs, f)


    def win(self):
        for i in self.logs:
            i.update({'result':1})
        self.log_turns()


    def lose(self):
        for i in self.logs:
            i.update({'result':0})
        # self.log_turns()


    def load_models(self):
        try:
            self.model = models.load_model(self.path + '/agent_{0}/squeezenet'.format(self.a_id))
        except:
            self.use_model = False


    def get_model_prediction(self):
        pass
        img_p = glob.glob(self.path + 'images/last_img.jpeg')[0]
        img = Image.open(img_p)
        # img = img.resize((128, 128))
        img = np.array(img)

        img = img[int(self.rect.centerx) - 64:int(self.rect.centerx) + 64,int(self.rect.centery) - 64:int(self.rect.centery) + 64,:]

        img = img.astype(np.float32)
        img /= 255
        x = self.model.predict(np.array([img]))
        # print(x)
        return x


def train_models(path, a_id, t_id):
    results = glob.glob(path + 'agent_{0}/*.json'.format(a_id))
    # print(results)

    x, y1, y2 = [], [], []
    for i in results:
        try:
            with open(i, 'r') as f:
                # print(i)
                j = json.load(f)
                # print(len(j))
                if len(j) > max_record_len:
                    continue
                for k in j:
                    img_p = glob.glob(path + 'images/g_img_{0}_{1}_{2}.jpeg'.format(re.findall('\d+', i)[-2], k['move'], t_id))[0]
                    img = Image.open(img_p)
                    # img = img.resize((128, 128))
                    img = np.array(img)
                    # print()
                    img = img[int(k['x']) - 64:int(k['x']) + 64, int(k['y']) - 64:int(k['y']) + 64, :]
                    img = img.astype(np.float32)
                    img /= 255

                    y1_t = int(k['m_index'])
                    y2_t = int(k['s_index'])
                    y1_t2 = [0 for _ in range(option_space_size)]
                    y2_t2 = [0 for _ in range(option_space_size)]

                    y1_t2[y1_t] = 1
                    y2_t2[y2_t] = 1
                    y1_t2 = np.array(y1_t2)
                    y2_t2 = np.array(y2_t2)

                    x.append(img)
                    y1.append(y1_t2)
                    y2.append(y2_t2)

        except:
            print(i)
            traceback.print_exc()

    x = np.array(x)
    y1 = np.array(y1)
    y2 = np.array(y2)

    x_train, x_val, y1_train, y1_val, y2_train, y2_val = train_test_split(x, y1, y2, test_size=.1)

    print(x_train.shape)
    net = get_net()
    cb1 = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
    cb2 = callbacks.ModelCheckpoint(path + 'agent_{0}/squeezenet'.format(a_id), monitor='val_loss', verbose=0, save_best_only=True,
                                    save_weights_only=False, mode='auto', period=1)
    net.fit(x_train, [y1_train, y2_train], validation_data=(x_val, [y1_val, y2_val]), epochs=10, callbacks=[cb1, cb2], batch_size=16)



class Board():

    def __init__(self, path = None, g_id = 0):
        self.team1 = []
        self.team2 = []
        self.team1.append(Agent(team = 1, a_id = 1, bounds=(16, 47, 16, 47), x = random.randint(16, 47), y = random.randint(16, 47), color=(255, 1, 1), g_id = g_id, path=path, range = 16, kill_angle=.1, use_model = random.choice([True, True])))
        self.team1.append(Agent(team = 1, a_id = 2, bounds=(16, 47, 16, 47), x = random.randint(16, 47), y = random.randint(16, 47), color=(255, 50, 1), g_id = g_id, path=path, range = 16, kill_angle=.1, use_model = random.choice([True, True])))
        self.team1.append(Agent(team = 1, a_id = 3, bounds=(16, 47, 16, 47), x = random.randint(16, 47), y = random.randint(16, 47), color=(255, 100, 1), g_id = g_id, path=path, range = 16, kill_angle=.1, use_model = random.choice([True, True])))
        self.team1.append(Agent(team = 1, a_id = 4, bounds=(16, 47, 16, 47), x = random.randint(16, 47), y = random.randint(16, 47), color=(255, 150, 1), g_id = g_id, path=path, range = 16, kill_angle=.1, use_model = random.choice([True, True])))
        self.team1.append(Agent(team = 1, a_id = 5, bounds=(16, 47, 16, 47), x = random.randint(16, 47), y = random.randint(16, 47), color=(255, 200, 1), g_id = g_id, path=path, range = 16, kill_angle=.1, use_model = random.choice([True, True])))
        self.team2.append(Agent(team = 2, a_id = 6, bounds=(16, 47, 16, 47), x = random.randint(16, 47), y = random.randint(16, 47), color=(1, 1, 255), g_id = g_id, path=path, range = 16, kill_angle=.1, use_model = random.choice([False, False])))
        self.team2.append(Agent(team = 2, a_id = 7, bounds=(16, 47, 16, 47), x = random.randint(16, 47), y = random.randint(16, 47), color=(1, 50, 255), g_id = g_id, path=path, range = 16, kill_angle=.1, use_model = random.choice([False, False])))
        self.team2.append(Agent(team = 2, a_id = 8, bounds=(16, 47, 16, 47), x = random.randint(16, 47), y = random.randint(16, 47), color=(1, 100, 255), g_id = g_id, path=path, range = 16, kill_angle=.1, use_model = random.choice([False, False])))
        self.team2.append(Agent(team = 2, a_id = 9, bounds=(16, 47, 16, 47), x = random.randint(16, 47), y = random.randint(16, 47), color=(1, 150, 255), g_id = g_id, path=path, range = 16, kill_angle=.1, use_model = random.choice([False, False])))
        self.team2.append(Agent(team = 2, a_id = 10, bounds=(16, 47, 16, 47), x = random.randint(16, 47), y = random.randint(16, 47), color=(1, 200, 255), g_id = g_id, path=path, range = 16, kill_angle=.1, use_model = random.choice([False, False])))



    def render_agents(self):
        return [i for i in self.team1 + self.team2 if i.alive]


    def run_turn(self, team):
        shots = []

        if team == 1:
            for i in self.team1:
                res = i.run_turn()

                if 'shot' in res:
                    for j in self.team1:
                        j.get_shot_at(res['shot'])
                    for j in self.team2:
                        j.get_shot_at(res['shot'])

                    shots.append(res['shot']['segment_scaled'])

        if team == 2:
            for i in self.team2:
                res = i.run_turn()

                if 'shot' in res:
                    for j in self.team1:
                        j.get_shot_at(res['shot'])
                    for j in self.team2:
                        j.get_shot_at(res['shot'])
                    shots.append(res['shot']['segment_scaled'])
        return shots


    def check_if_game_over(self):
        if len([i for i in self.team1 if i.alive]) == 0:
            for j in self.team1:
                j.lose()
            for j in self.team2:
                j.win()

        elif len([i for i in self.team2 if i.alive]) == 0:
            for j in self.team2:
                j.lose()
            for j in self.team1:
                j.win()
        else:
            return False
        return True

    def get_result(self):
        if len([i for i in self.team1 if i.alive]) == 0:
            return 2
        elif len([i for i in self.team2 if i.alive]) == 0:
            return 1
        else:
            return 0


def main():
    path = '/home/td/Documents/rl_tests/swarm_1/dual/'

    if not os.path.exists(path + '/images/'):
        os.makedirs(path + '/images/')

    pygame.init()
    screen = pygame.display.set_mode((512, 512))
    pygame.display.set_caption('test')
    pygame.mouse.set_visible(0)
    screen.fill((0, 0, 0))

    st = time.time()
    game_count = 10000
    try:
        result_dicts = pd.read_csv('res.csv').to_dict(orient='records')
    except:
        result_dicts = []
    game_count_start = 275

    g = game_count_start

    result_dict = {}


    while g < game_count:
        b = Board(path = path, g_id = g)
        screen.fill((0, 0, 0))
        s = b.render_agents()
        for sprite_one in s:
            screen.blit(sprite_one.image, (sprite_one.rect.centerx,sprite_one.rect.centery))
        pygame.display.flip()

        files_to_remove = glob.glob(path + '/images/' + "g_img_{0}_{1}.jpeg".format(g, '*'))
        for i in files_to_remove:
            os.remove(i)

        count = 0
        while not b.check_if_game_over() and count < max_record_len:
            for turn in [1, 2]:
                gc.collect()
                pygame.image.save(screen, path + '/images/' + "last_img.jpeg")
                pygame.image.save(screen, path + '/images/' + "g_img_{0}_{1}_{2}.jpeg".format(g, count, turn))
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
        if b.check_if_game_over():
            g += 1
            print(g, time.time() - st)
            result = b.get_result()
            result_dict.setdefault(result, 0)
            result_dict[result] += 1
            print('results', result_dict)
            result_dicts.append({'g':g, 'result':result})
            df = pd.DataFrame.from_dict(result_dicts)
            df.to_csv('res.csv', index=False)
            del b
            if g % 100 == 0 and g > 1:
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







if __name__ == '__main__':

    main()
