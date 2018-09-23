#naive solution, training on games won with random policy

import os, sys
import pygame
from pygame.locals import *
import math
import random
import numpy as np
import time
import tensorflow
from keras import layers, models, optimizers, callbacks
from keras import backend as K
import glob
import json
import shutil
from PIL import Image
import re
from sklearn.model_selection import train_test_split

max_record_len = 500

def fire_module(x, fire_id, squeeze=16, expand=64):
    sq1x1 = "squeeze1x1"
    exp1x1 = "expand1x1"
    exp3x3 = "expand3x3"
    relu = "relu_"
    #https://github.com/rcmalli/keras-squeezenet/blob/master/keras_squeezenet/squeezenet.py
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = layers.Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = layers.Activation('relu', name=s_id + relu + sq1x1)(x)
    left = layers.Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = layers.Activation('relu', name=s_id + relu + exp1x1)(left)
    right = layers.Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = layers.Activation('relu', name=s_id + relu + exp3x3)(right)
    x = layers.concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x


def get_squeeze_net():
    img_input = layers.Input((256, 256, 3))
    x = layers.Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(img_input)
    x = layers.Activation('relu', name='relu_conv1')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)
    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)
    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)
    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    x = layers.Dropout(0.5, name='drop9')(x)
    x = layers.Convolution2D(4, (1, 1), padding='valid', name='conv10')(x)
    x = layers.Activation('relu', name='relu_conv10')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation('tanh', name='loss')(x)
    model = models.Model(img_input, x, name='squeezenet')

    model.compile('adam', loss='mean_squared_error', metrics = ['mean_squared_error'])
    return model


class Agent(pygame.sprite.Sprite):
    def __init__(self, x, y, bounds, range = 12, kill_angle = .01, team = None, a_id = None, alive = True,
                 color = None, pixels_per_square = 8, path = None, g_id = 0):
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


    #Main function
    def run_turn(self):
        print('turn for agent', self.a_id, self.x, self.y)

        result = {}

        if self.alive:
            #TODO: replace random with 3 networks
            x_m = random.random() * 2 - 1
            y_m = random.random() * 2 - 1
            a_m_t = math.atan2(y_m, x_m)
            a_m = math.atan2(y_m, x_m)/math.pi
            x_m = math.cos(a_m_t)
            y_m = math.sin(a_m_t)

            self.move(x_m, y_m)
            x_s = random.random() * 2 - 1
            y_s = random.random() * 2 - 1
            a_s = math.atan2(y_s, x_s)/math.pi

            result['shot'] = self.shoot(x_s, y_s)
            self.logs.append({'x_m':x_m, 'y_m':y_m, 'x_s':x_s, 'y_s':y_s, 'move':self.move_count})
            self.move_count += 1
            # self.log_turns()
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


    # def build_models(self):
    #     self.m_d = pass



def train_models(path, a_id):
    results = glob.glob(path + 'agent_{0}/*.json'.format(a_id))
    print(results)

    x, y = [], []
    for i in results:
        with open(i, 'r') as f:
            j = json.load(f)
            print(len(j))
            if len(j) > 200:
                continue
            for k in j:
                img_p = glob.glob(path + 'images/g_img_{0}_{1}.jpeg'.format(re.findall('\d+', i)[-2], k['move']))[0]
                img = np.array(Image.open(img_p))
                x.append(img)
                y.append(np.array([k['x_m'], k['y_m'], k['x_s'], k['y_s']]))

    x = np.array(x)
    y = np.array(y)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.1)
    net = get_squeeze_net()
    cb = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
    net.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, callbacks=[cb], batch_size=512)



class Board():

    def __init__(self, path = None, g_id = 0):
        self.team1 = []
        self.team2 = []
        self.team1.append(Agent(team = 1, a_id = 1, bounds=(0, 31, 0, 31), x = 2, y = 2, color=(255, 0, 0), g_id = g_id, path=path))
        self.team2.append(Agent(team = 2, a_id = 2, bounds=(0, 31, 0, 31), x = 29, y = 29, color=(0, 0, 255), g_id = g_id, path=path))


    def render_agents(self):
        return self.team1 + self.team2


    def run_turn(self):
        shots = []

        for i in self.team1:
            res = i.run_turn()

            if 'shot' in res:
                for j in self.team2:
                    j.get_shot_at(res['shot'])

                shots.append(res['shot']['segment_scaled'])


        for i in self.team2:
            res = i.run_turn()

            if 'shot' in res:
                for j in self.team1:
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


def main():
    path = '/home/td/Documents/rl_tests/swarm_1/dual/'

    if not os.path.exists(path + '/images/'):
        os.makedirs(path + '/images/')

    pygame.init()
    screen = pygame.display.set_mode((256, 256))
    pygame.display.set_caption('test')
    pygame.mouse.set_visible(0)
    screen.fill((0, 0, 0))

    for g in range(10000):
        b = Board(path = path, g_id = g)
        s = b.render_agents()
        for sprite_one in s:
            screen.blit(sprite_one.image, (sprite_one.rect.centerx,sprite_one.rect.centery))
        pygame.display.flip()

        count = 0
        while not b.check_if_game_over() and count < max_record_len:
            shots = b.run_turn()
            pygame.image.save(screen, path + '/images/' + "g_img_{0}_{1}.jpeg".format(g, count))
            count += 1
            screen.fill((0, 0, 0))
            for s in shots:
                print(s)
                pygame.draw.line(screen, (0, 255, 0), s[0], s[1])
            s = b.render_agents()
            for sprite_one in s:
                screen.blit(sprite_one.image, (sprite_one.rect.centerx, sprite_one.rect.centery))
            pygame.display.flip()
            # time.sleep(1)
        # time.sleep(5)


if __name__ == '__main__':
    # main()
    train_models('/home/td/Documents/rl_tests/swarm_1/dual/', 1)
