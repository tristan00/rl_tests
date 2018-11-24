import lightgbm as lgb
import random
import numpy as np
from sklearn.model_selection import train_test_split
import copy
import traceback
import pandas as pd
from keras import layers, models, optimizers, callbacks
from keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestRegressor
import pickle

elo_d = 500
elo_k = elo_d/10
chance_to_stop = .01
max_iter =  100
forward_view = 10
max_training_size = 100000

g_preset = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0,
      0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0,
      0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0,
      0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1,
      0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0,
      0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
      0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0,
      0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1,
      1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0,
      0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1,
      0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1,
      1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1,
      0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0,
      0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1,
      1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1,
      0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
      1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0,
      1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
      1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1,
      0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
      0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0,
      1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1,
      1]

lgbm_params =  {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'l1',
    "learning_rate": 0.01,
    "max_depth": -1,
    'num_leaves':127
    }

path = r'C:\Users\trist\Documents\prisoner_models/'
epsilon = .001
epsilon_decay = .1
epsilon_decay_period = 10
min_epsilon = .001
maximum_elo = 10000
minimum_elo = 10
starting_elo = 1000

def calculate_new_elo(outcome, player_1_elo, player_2_elo):
    expected_outcome = 1 / (1 + 10**((player_2_elo - player_1_elo)/elo_d))
    new_elo =  player_1_elo + (elo_k * (outcome-expected_outcome))
    return min(maximum_elo, max(minimum_elo, new_elo))


class DBot():
    def __init__(self, n, history_len = 5, model_depth = 1, base_alg = 'random', trainable = False):
        self.b_id = n
        self.base_alg = base_alg
        self.history_len = history_len
        self.games = 0
        self.total_score = 0
        self.model_depth = model_depth
        self.trainable = trainable
        self.x = []
        self.y = []
        self.elo = starting_elo
        self.flag_1 = 0
        self.move_count = 0

        self.feature_names = ['agent_move_{0}_moves_past'.format(i) for i in range(self.history_len - 1, -1, -1)] + \
        ['opponent_move_{0}_moves_past'.format(i) for i in range(self.history_len, 0, -1)] + ['next_move']
        # print(self.feature_names)

        try:
            raise Exception()
            self.model = lgb.Booster(model_file=path + '/lgbmodel_{0}'.format(self.b_id))
            self.use_model = True
        except:
            # traceback.print_exc()
            self.model = None
            self.use_model = False


    def save_model(self):
        with open(path + '/agent_{0}.pkl'.format(self.b_id), 'wb') as f:
            pickle.dump(self.model, f)
        export_graphviz(self.model, out_file=path + '/agent_{0}.txt'.format(self.get_id()), feature_names=self.feature_names)


    def get_id(self):
        if self.trainable:
            return str(self.b_id) + '_' + self.base_alg + '_' + 'trainable' + '_model_depth_' + str(self.model_depth)
        else:
            return str(self.b_id) + '_' + self.base_alg + '_' + 'not_trainable'


    def train_dnn(self):
        if len(self.x) > 0:

            x = np.array(self.x)
            y = np.array(self.y)

            self.scaler = StandardScaler()
            x = x.astype(np.float32)
            x = self.scaler.fit_transform(x)
            train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=.01, random_state=1)

            m_input = layers.Input(shape=(9,), name = 'input')
            mx = layers.Dense(1028, activation='relu', name='dl1')(m_input)
            mx = layers.Dense(1028, activation='relu', name='dl2')(mx)
            mx = layers.Dense(1, activation='relu', name='predictions1')(mx)
            model = models.Model(inputs=m_input, outputs=mx, name='dnn')
            model.compile('adam', loss='mean_squared_error')

            if train_x.shape[0] < 2000:
                chunk_size = 8
            elif train_x.shape[0] < 4000:
                chunk_size = 16
            elif train_x.shape[0] < 8000:
                chunk_size = 32
            elif train_x.shape[0] < 16000:
                chunk_size = 64
            elif train_x.shape[0] < 32000:
                chunk_size = 128
            elif train_x.shape[0] < 64000:
                chunk_size = 256
            elif train_x.shape[0] < 128000:
                chunk_size = 512
            elif train_x.shape[0] < 256000:
                chunk_size = 1024
            elif train_x.shape[0] < 512000:
                chunk_size = 2048
            else:
                chunk_size = 4096

            cb1 = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
            cb2 = callbacks.ModelCheckpoint(path + '/agent_{0}_net.h5'.format(self.b_id), monitor='val_loss', verbose=0,
                                            save_best_only=True,
                                            save_weights_only=False, mode='auto', period=1)
            model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=200,
                    callbacks=[cb1, cb2], batch_size=chunk_size)
            self.dnn = models.load_model(path + '/agent_{0}_net.h5'.format(self.b_id))
            self.use_model = True




    def train_model(self):
        if len(self.x) > 0 and self.trainable:
            print('training model')
            x = np.array(self.x)
            y = np.array(self.y)


            if x.shape[0] > max_training_size:
                print('agent {0} sampling'.format(self.b_id))
                x, _, y, _ = train_test_split(x, y, train_size = max_training_size)

            # self.model = RandomForestRegressor(max_depth=self.model_depth, n_estimators=10)
            self.model = DecisionTreeRegressor(max_depth=self.model_depth)
            self.model.fit(x, y)
            # train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=.01, random_state=1)
            # lgtrain = lgb.Dataset(train_x, train_y)
            # lgvalid = lgb.Dataset(val_x, val_y,)
            #
            # self.model = lgb.train(lgbm_params,
            # lgtrain,
            # num_boost_round=max_iter,
            # valid_sets=[lgtrain, lgvalid],
            # valid_names=['train', 'valid'],
            # early_stopping_rounds=10,
            # verbose_eval=-1
            # )
            #
            # self.model.save_model(path + '/lgbmodel_{0}'.format(self.b_id), num_iteration=self.model.best_iteration)
            # self.model = lgb.Booster(model_file=path + '/lgbmodel_{0}'.format(self.b_id))
            self.use_model = True
            self.save_model()



    def give_feedback(self,  move_history1, move_history2, move, score):
        small_move1_history = move_history1[-self.history_len:]
        small_move2_history = move_history2[-self.history_len:]

        while len(small_move1_history) < self.history_len:
            small_move1_history.insert(0, -1)
        while len(small_move2_history) < self.history_len:
            small_move2_history.insert(0, -1)



        new_x = small_move1_history + small_move2_history + [move]
        self.x.append(new_x)
        self.y.append(score)


    def predict_move(self, b1_move_history, b2_move_history, opponent_id = 0, move_num = 0, l_epsilon = 1.0):
        # print(move_history, score_history)

        if not self.use_model or random.random() < l_epsilon:
            if self.base_alg == 'random':
                return random.randint(0, 1)
            if self.base_alg == 'tit_for_tat':
                if b2_move_history and b2_move_history[-1] == 1:
                    return 1
                else:
                    return 0
            if self.base_alg == 'defect':
                return 1
            if self.base_alg == 'cooperate':
                return 0
            if self.base_alg == 'tit_for_2tat':
                if len(b1_move_history) >= 2 and b2_move_history[-1] == 1 and b2_move_history[-2] == 1:
                    return 1
                else:
                    return 0

            if self.base_alg == 'preset':
                return g_preset[self.move_count]

            if self.base_alg == 'tester':
                pass


            if self.base_alg == 'tranquilizer':
                pass
        else:

            small_move_history = b1_move_history[-self.history_len:]
            small_score_history = b2_move_history[-self.history_len:]

            while len(small_move_history) < self.history_len:
                small_move_history.insert(0, -1)
            while len(small_score_history) < self.history_len:
                small_score_history.insert(0, -1)

            x1 = small_move_history + small_score_history + [1]
            x0 = small_move_history + small_score_history + [0]

            # dnn_input1 = self.scaler.transform(np.array([x1]).astype(np.float32))
            # dnn_input0 = self.scaler.transform(np.array([x0]).astype(np.float32))
            #
            # x1_val = self.dnn.predict(dnn_input1)
            # x0_val = self.dnn.predict(dnn_input0)

            x1_val = self.model.predict(np.array([x1]))
            x0_val = self.model.predict(np.array([x0]))

            # print(x0_val[0], x1_val[0])

            if x1_val[0] > x0_val[0]:
                return 1
            elif x1_val[0] < x0_val[0]:
                return 0
            else:
                return random.randint(0,1)



class Game():
    def __init__(self, b1, b2, epsilon, retraining_frequency = .1, learning = True, max_rounds = 250):

        move1_history = []
        move2_history = []
        score1_history = []
        score2_history = []
        input1_1_history = []
        input2_1_history = []
        input1_2_history = []
        input2_2_history = []
        self.learning = learning

        b1.flag1 = 0
        b2.flag2 = 0
        b1.move_count = 0
        b2.move_count = 0

        game_count = 0
        for game_count in range(max_rounds):
            # b1_move = b1.predict_move(move1_history, score1_history, l_epsilon = epsilon)
            # b2_move = b2.predict_move(move2_history, score2_history, l_epsilon = epsilon)

            b1_move = b1.predict_move(move1_history, move2_history, l_epsilon = epsilon)
            b2_move = b2.predict_move(move2_history, move1_history, l_epsilon = epsilon)

            input1_1_history.append(copy.deepcopy(move1_history))
            input1_2_history.append(copy.deepcopy(score1_history))

            input2_1_history.append(copy.deepcopy(move2_history))
            input2_2_history.append(copy.deepcopy(score2_history))

            if b1_move == 1 and b2_move == 1:
                # b1.give_feedback(move1_history, score1_history, b1_move, 1)
                # b2.give_feedback(move2_history, score2_history, b2_move, 1)
                score1_history.append(1)
                score2_history.append(1)

            elif b1_move == 1 and b2_move == 0:
                # b1.give_feedback(move1_history, score1_history, b1_move, 5)
                # b2.give_feedback(move2_history, score2_history, b2_move, 0)
                score1_history.append(5)
                score2_history.append(0)

            elif b1_move == 0 and b2_move == 1:
                # b1.give_feedback(move1_history, score1_history, b1_move, 0)
                # b2.give_feedback(move2_history, score2_history, b2_move, 5)
                score1_history.append(0)
                score2_history.append(5)

            elif b1_move == 0 and b2_move == 0:
                # b1.give_feedback(move1_history, score1_history, b1_move, 3)
                # b2.give_feedback(move2_history, score2_history, b2_move, 3)
                score1_history.append(3)
                score2_history.append(3)

            move1_history.append(b1_move)
            move2_history.append(b2_move)

            b1.move_count += 1
            b2.move_count += 1

            if random.random() < chance_to_stop and learning:
                break

        self.move1_history = move1_history
        self.move2_history = move2_history
        s1 = sum(score1_history)/len(score1_history)
        s2 = sum(score2_history)/len(score2_history)


        # #competitive
        # if s1 > s2:
        #     s1_2 = 1
        #     s2_2 = 0
        # elif s1 < s2:
        #     s1_2 = 0
        #     s2_2 = 1
        # else:
        #     s1_2 = 0
        #     s2_2 = 0


        for count, (i1, i2, i3, i4, i5, i6, i7, i8) in enumerate(zip(move1_history, move2_history, score1_history, score2_history, input1_1_history, input2_1_history, input1_2_history, input2_2_history)):

            fs1 = sum(score1_history[count:count+b1.history_len])/len(score1_history[count:count+b1.history_len])
            fs2 = sum(score2_history[count:count+b2.history_len])/len(score2_history[count:count+b2.history_len])
            b1.give_feedback(i5, i6, i1, fs1)
            b2.give_feedback(i6, i5, i2, fs2)

        # print(move1_history)
        # print(score1_history)
        # print(move2_history)
        # print(score2_history)

        if random.random() < retraining_frequency and learning:
            b1.train_model()
            b2.train_model()

            # b1.train_dnn()
            # b2.train_dnn()

        b1_score_final = sum(score1_history)/len(score1_history)
        b2_score_final = sum(score2_history)/len(score2_history)

        if learning:
            b1_new_elo = calculate_new_elo(b1_score_final / (b2_score_final + b1_score_final), b1.elo, b2.elo)
            b2_new_elo = calculate_new_elo(b2_score_final / (b2_score_final + b1_score_final), b2.elo, b1.elo)

            b1.elo = b1_new_elo
            b2.elo = b2_new_elo

        print('total games', game_count, 'scores:', b1_score_final, b2_score_final, 'bots:',b1.get_id(),b2.get_id(),'elos:', b1.elo, b2.elo, 'retraining_frequency:', retraining_frequency)

        # if learning and sum(score1_history)/len(score1_history) > sum(score2_history)/len(score2_history):
        #     b1.elo = calculate_new_elo(1, b1.elo, b2.elo)
        #     b2.elo = calculate_new_elo(0, b1.elo, b2.elo)
        # elif learning and sum(score1_history)/len(score1_history) < sum(score2_history)/len(score2_history):
        #     b1.elo = calculate_new_elo(0, b1.elo, b2.elo)
        #     b2.elo = calculate_new_elo(1, b1.elo, b2.elo)
        self.score = b1_score_final, b2_score_final






def analyze_character(a, b):
    uncalled_aggression = 0
    retaliation = 0
    forgiveness = 0
    first_move = a[0]

    for i in range(len(a)):
        if i >= 3:
            if max(b[i-3:i]) == 0 and a[i] == 1:
                uncalled_aggression += 1
        if i > 1:
            if b[i-1] == 1 and a[i] == 1 and a[i-1] == 0:
                retaliation += 1
        if i > 5:
            if b[i-3] == 1 and a[i-2] == 1 and b[i-2] == 0 and a[i-1] == 0:
                forgiveness += 1
            elif b[i - 3] == 1 and a[i - 2] == 1 and b[i - 1] == 0 and a[i] == 0:
                forgiveness += 1
    return uncalled_aggression/len(a), retaliation/len(a), forgiveness/len(a), first_move


def get_character(b):
    n_bot = DBot(1, history_len=100, model_depth=1, base_alg='preset', trainable=False)
    g = Game(b, n_bot, 0.0, retraining_frequency =0, learning=False, max_rounds=500)
    return analyze_character(g.move1_history, g.move2_history)


def rate_bots_comparative(bots):
    res_array = np.zeros((len(bots), len(bots)))

    df = pd.DataFrame(data = res_array,
                      index = [i.get_id() for i in bots],
                      columns=[i.get_id() for i in bots])
    ratings = []

    for count1, b1 in enumerate(bots):
        average = 0
        for count2, b2 in enumerate(bots):
            if b1.b_id != b2.b_id:
                print(b1.b_id, b2.b_id)
                g1 = Game(b1, b2, 0.0, learning=False)
                df.loc[b1.get_id(), b2.get_id()] = g1.score[0]
                # res_array[b2.b_id, b1.b_id] = g1.score[1]
                average += g1.score[0]
        average = average / (len(bots)-1)
        ratings.append({'bot':b1, 'average':average})


    df['elo'] = 0

    for i in bots:
        df.loc[i.get_id(), 'elo'] = i.elo


    df['model_depth'] = 0

    for i in bots:
        df.loc[i.get_id(), 'model_depth'] = i.model_depth

    df['uncalled_aggression'] = 0
    df['retaliation'] = 0
    df['forgiveness'] = 0
    df['first_move'] = 0

    for i in bots:
        char_res = get_character(i)
        df.loc[i.get_id(), 'uncalled_aggression'] = char_res[0]
        df.loc[i.get_id(), 'retaliation'] = char_res[1]
        df.loc[i.get_id(), 'forgiveness'] = char_res[2]
        df.loc[i.get_id(), 'first_move'] = char_res[3]


    df['average'] = df[[i.get_id() for i in bots]].mean(axis = 1)
    df['stddev'] = df[[i.get_id() for i in bots]].std(axis = 1)
    df['median'] = df[[i.get_id() for i in bots]].median(axis = 1)


    # df = pd.DataFrame(data = res_array,
    #                   index = [i.get_id() for i in bots],
    #                   columns=[i.get_id() for i in bots])
    return df, ratings


# for g_count in range(16,17):
#     # bots = [DBot(i, history_len=2, model_depth=8) for i in range(g_count)]

g_count = 4
bots = []
bots.append(DBot(0, history_len=50, model_depth=1, base_alg = 'tit_for_tat', trainable = False))
bots.append(DBot(1, history_len=50, model_depth=1, base_alg = 'random', trainable = False))
bots.append(DBot(2, history_len=50, model_depth=1, base_alg='defect', trainable=False))
bots.append(DBot(3, history_len=50, model_depth=1, base_alg='cooperate', trainable=False))
bots.append(DBot(4, history_len=50, model_depth=1, base_alg='tit_for_2tat', trainable=False))

for i in range(g_count):
    bots.append(DBot(i+5, history_len=50, model_depth=random.randint(3, 10), base_alg='random', trainable=True))


max_count = g_count + 5
scores = []
score_list = []

base_retraining_frequency = .2
generations = 1000

for g in range(generations):

    for i in range(100000):
        random.shuffle(bots)
        b1 = bots[0]
        b2 = bots[1]
        print(i, epsilon, b1.b_id, b2.b_id)

        g = Game(b1, b2, epsilon, retraining_frequency = base_retraining_frequency*(.5**(i/5000)))
        scores.append({'s_{0}_score'.format(b1.b_id):g.score[0],
                       # 's_{0}_history_len'.format(b1.b_id):b1.history_len,
                       # 's_{0}_history_len'.format(b2.b_id):b1.history_len, 's_{0}_model_depth'.format(b1.b_id): b1.history_len,
                       # 's_{0}_model_depth'.format(b2.b_id): b1.history_len,
                       's_{0}_score'.format(b2.b_id):g.score[1]})
        df = pd.DataFrame.from_dict(scores)
        df.to_csv('res_{0}.csv'.format(g_count), index = False)

        if b1.b_id > 4 and b2.b_id > 4:
            score_list.append(g.score[0])
            score_list.append(g.score[1])

        if len(score_list) > 0:
            print('trailing results', sum(score_list[-1000:])/len(score_list[-1000:]), len(score_list[-1000:]))

        if i % epsilon_decay_period ==0 and i > 0:
            epsilon = max(epsilon * (1 - epsilon_decay), min_epsilon)

        if i%5000 == 0 and i > 0:
            comp_df, ratings = rate_bots_comparative(bots)
            comp_df.to_csv('comp_res_{0}.csv'.format(g))
            ratings.sort(key = lambda x: x['average'], reverse = True)

            print('generation : {0}'.format(g))
            for i in [ratings for b in ratings][:len(ratings)/2]:
                print('selecting bot: {0}'.format(i['bot'].get_id()))
            for i in [ratings for b in ratings][len(ratings)/2:]:
                print('regecting bot: {0}'.format(i['bot'].get_id()))


            bots = [ratings['bot'] for b in ratings][:len(ratings)/2]

            for b in range(len(ratings)/2):
                max_count += 1
                bots.append(DBot(max_count, history_len=50, model_depth=random.randint(3, 10), base_alg='random', trainable=True))
            break




comp_df = rate_bots_comparative(bots)
comp_df.to_csv('comp_res_final.csv')
# for i in bots:
#     print(i.b_id, rate_Bot(i))

