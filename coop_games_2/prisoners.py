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
import gc


elo_d = 500
elo_k = elo_d/10
chance_to_stop = .01
max_iter =  100
forward_view = 10
base_retraining_frequency = .2
generations = 1000
decay_period = 1000
generation_training_size = 20000
max_training_size = 50000
max_num_of_bots = 100
path = r'C:\Users\trist\Documents\prisoner_models\saved_games/'
sum_path =r'C:\Users\trist\Documents\prisoner_models\saved_data/'
epsilon = .001
epsilon_decay = .1
epsilon_decay_period = 100
min_epsilon = .25
maximum_elo = 10000
minimum_elo = 10
starting_elo = 1000
prob_of_trainable = 1.0
rating_prob = .5
max_tit = 3
max_tat = 3
base_alg_random_prob = 0.01
random_defect_chance = .125
max_model_depth = 10
min_model_depth = 5

survival_chance_limit = .8 #Scales survival rate of generaltion so scaling survival percentage only starts for the bottom n%

# base_algorithms =  ['tit_for_tat', 'random', 'defect', 'no_forgiveness', 'tit_for_2tat', 'cooperate']
base_algorithms =  ['{0}tit_for_{1}tat', 'random', 'defect', 'cooperate', 'no_forgiveness', 'tit_for_tat_first_defect',
                    'tit_for_tat_random_defect', 'tit_for_tat_random_defect', 'tester']
eff_base_algs = ['{0}tit_for_{1}tat'.format(i, j) for i in range(1, max_tit + 1)
                 for j in range(1, max_tat + 1) ] + ['random', 'defect', 'cooperate', 'no_forgiveness',
                                                     'tit_for_tat_first_defect', 'tit_for_tat_random_defect', 'tester']
max_data_size = 100000

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



def calculate_new_elo(outcome, player_1_elo, player_2_elo):
    expected_outcome = 1 / (1 + 10**((player_2_elo - player_1_elo)/elo_d))
    new_elo =  player_1_elo + (elo_k * (outcome-expected_outcome))
    return min(maximum_elo, max(minimum_elo, new_elo))


class DBot():
    def __init__(self, n, history_len = 5, model_depth = 1, base_alg = 'random', trainable = False, generation = 0, decay_period = 100, use_reputation = 0, base_alg_random_prob = 0, alg_constants = (1, 1)):
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
        self.generation = generation
        self.g_count = 0
        self.base_training_rate = base_retraining_frequency
        self.decay_period = decay_period
        self.epsilon = 1.0
        self.base_alg_random_prob = base_alg_random_prob
        if use_reputation == 1:
            self.use_reputation = True
        else:
            self.use_reputation = False

        self.alg_constants = alg_constants
        if base_alg == '{0}tit_for_{1}tat':
            self.alt_base_alg = self.base_alg.format(alg_constants[0], alg_constants[1])
        else:
            self.alt_base_alg = self.base_alg

        if self.use_reputation:
            rep_fetures = ['opponent_rep_first_move', 'opponent_rep_forgiveness', 'opponent_rep_retaliation', 'opponent_rep_uncalled_aggression', 'opponent_rep_score',
             'opponent_rep_use_reputation', 'opponent_rep_generation', 'opponent_rep_elo', 'opponent_rep_games', 'opponent_rep_model_depth']

            self.feature_names = ['agent_move_{0}_moves_past'.format(i) for i in range(self.history_len - 1, -1, -1)] + \
                                 ['opponent_move_{0}_moves_past'.format(i) for i in range(self.history_len, 0, -1)] + [
                                     'next_move'] + rep_fetures
        else:
            self.feature_names = ['agent_move_{0}_moves_past'.format(i) for i in range(self.history_len - 1, -1, -1)] + \
            ['opponent_move_{0}_moves_past'.format(i) for i in range(self.history_len, 0, -1)] + ['next_move']
            # print(self.feature_names)

        self.model = None
        self.use_model = False
        self.trained_max = False

        #Characteristics:
        self.first_move = -1
        self.forgiveness = -1
        self.retaliation = -1
        self.uncalled_aggression = -1
        self.score = -1


    def get_reputation(self):
        return [self.first_move, self.forgiveness, self.retaliation, self.uncalled_aggression, self.score, self.use_reputation, self.generation, self.elo, self.games, self.model_depth]


    def save_model(self):
        with open(path + '/agent_{0}.pkl'.format(self.b_id), 'wb') as f:
            pickle.dump(self.model, f)
        export_graphviz(self.model, out_file=path + '/agent_{0}.txt'.format(self.get_id()), feature_names=self.feature_names)


    def get_id(self):
        if self.trainable or self.use_model:
            if self.use_reputation:
                return str(self.b_id) + '_' + 'generation_' + str(self.generation) + '_' + self.alt_base_alg + '_' + 'trainable' + '_model_depth_' + str(self.model_depth) + '_using_reputation'
            else:
                return str(self.b_id) + '_' + 'generation_' + str(self.generation) + '_' + self.alt_base_alg + '_' + 'trainable' + '_model_depth_' + str(self.model_depth) + '_not_using_reputation'

        else:
            return str(self.b_id) + '_' + self.alt_base_alg + '_' + 'not_trainable'



    def train_model(self, always = False, final = False):

        training_prob = self.base_training_rate * (.5 ** (self.g_count / self.decay_period))
        # print(len(self.x), training_prob, self.trained_max)
        # print('training_prob', training_prob)


        if (len(self.x) > 0 and self.trainable and (random.random() < training_prob or always) and not self.trained_max)\
                or (not self.trained_max and len(self.x) >= max_data_size):
            # print('training model')
            x = np.array(self.x)
            y = np.array(self.y)

            if x.shape[0] > max_training_size:
                print('agent {0} sampling'.format(self.b_id))
                sample_portion = 1- max_training_size/x.shape[0]
                x, _, y, _ = train_test_split(x, y, test_size = sample_portion)

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

            if not self.trained_max and (len(self.x) >= max_data_size or final):
                print(self.get_id(), 'model training done', len(self.x))
                self.trained_max = True


    def give_feedback(self,  move_history1, move_history2, move, score, opponent_reputation):
        if len(self.x) < max_data_size and not self.trained_max:
            small_move1_history = move_history1[-self.history_len:]
            small_move2_history = move_history2[-self.history_len:]

            while len(small_move1_history) < self.history_len:
                small_move1_history.insert(0, -1)
            while len(small_move2_history) < self.history_len:
                small_move2_history.insert(0, -1)

            if self.use_reputation:
                new_x = small_move1_history + small_move2_history + [move] + opponent_reputation
            else:
                new_x = small_move1_history + small_move2_history + [move]

            self.x.append(new_x)
            self.y.append(score)



    def predict_move(self, b1_move_history, b2_move_history, rep, opponent_id = 0, move_num = 0, l_epsilon = 1.0):
        # print(move_history, score_history)

        epsilon = self.epsilon * ((1 - epsilon_decay)**self.g_count)
        # print('epsilon', epsilon)

        if self.use_model and random.random() > epsilon:
            small_move_history = b1_move_history[-self.history_len:]
            small_score_history = b2_move_history[-self.history_len:]

            while len(small_move_history) < self.history_len:
                small_move_history.insert(0, -1)
            while len(small_score_history) < self.history_len:
                small_score_history.insert(0, -1)

            if self.use_reputation:
                x1 = small_move_history + small_score_history + [1] + rep
                x0 = small_move_history + small_score_history + [0] + rep
            else:
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
            # else:
            #     return random.randint(0, 1)

        if random.random() < self.base_alg_random_prob and self.trainable:
            return random.randint(0, 1)
        if self.base_alg == 'random':
            return random.randint(0, 1)

        if self.base_alg == '{0}tit_for_{1}tat':
            # print('start')
            # print(self.alg_constants)
            # print(b1_move_history[-5:])
            # print(b2_move_history[-5:])
            # print(len( b2_move_history[-self.alg_constants[1]-0:-0]))
            #
            # if sum(b2_move_history[-3:]) == 3 and self.alg_constants[1] == 3:
            #     print('here')

            for i in range(self.alg_constants[0]):
                if i == 0:
                    b_sub = b2_move_history[-self.alg_constants[1]:]
                else:
                    b_sub = b2_move_history[-self.alg_constants[1]-i:-i]
                # print(b2_move_history[-self.alg_constants[1]-i:-i])
                # print(i)
                # print(self.alg_constants[1])
                # print(len(b_sub))
                # print(sum(b_sub))
                # print(len( b2_move_history[:-i]))
                # print(len( b2_move_history[:-i][-self.alg_constants[1]-i:]))
                # print(len( b2_move_history[:-i][-self.alg_constants[1]-i:]) >= self.alg_constants[1])
                # print(len(b_sub) == sum(b_sub))

                if len( b_sub) >= self.alg_constants[1] and len(b_sub) == sum(b_sub):
                    # print('pred 1')
                    return 1
            # print('pred 0')
            return 0

            #
            # if len(b1_move_history) > 3:
            #     print('here')
            # for i in range(0, self.alg_constants[0]):
            #
            #     if len(b1_move_history) >= self.alg_constants[1] + i:
            #         print('here')
            #         print(i)
            #         print(b1_move_history[-self.alg_constants[1] - i:-i])
            #         print(len(b1_move_history[-self.alg_constants[1] - i:-i]))
            #         print(sum(b1_move_history[-self.alg_constants[1] - i:-i]) )
            #
            #     if len(b1_move_history) >= self.alg_constants[1] + i and \
            #             sum(b1_move_history[-self.alg_constants[1] - i:-i]) == len(b1_move_history[-self.alg_constants[1] - i:-i]):
            #         return 1
            # return 0


        if self.base_alg == 'tit_for_tat':
            if b2_move_history and b2_move_history[-1] == 1:
                return 1
            else:
                return 0
        if self.base_alg == 'tit_for_2tat':
            if len(b1_move_history) >= 2 and b2_move_history[-1] == 1 and b2_move_history[-2] == 1:
                return 1
            else:
                return 0

        if self.base_alg == 'defect':
            return 1
        if self.base_alg == 'cooperate':
            return 0


        if self.base_alg == 'preset':
            return g_preset[self.move_count]

        if self.base_alg == 'no_forgiveness':
            if b2_move_history and b2_move_history[-1] == 1:
                self.flag_1 = 1
            if self.flag_1 == 1:
                return 1
            else:
                return 0

        if self.base_alg == 'tit_for_tat_first_defect':
            if (b2_move_history and b2_move_history[-1] == 1) or not b2_move_history:
                return 1
            else:
                return 0

        if self.base_alg == 'tit_for_tat_random_defect':
            if (b2_move_history and b2_move_history[-1] == 1) or random.random() < random_defect_chance:
                return 1
            else:
                return 0

        if self.base_alg == 'tester':
            if (b2_move_history and b2_move_history[-1] == 1 and b1_move_history[-1] == 0) or \
                    (b2_move_history and b2_move_history[-1] == 0  and b1_move_history[-1] == 1) or not b2_move_history:
                return 1
            else:
                return 0

        if self.base_alg == 'tranquilizer':
            pass





class Game():
    def __init__(self, b1, b2, learning = True, max_rounds = 250):


        b1.g_count += 1
        b2.g_count += 1

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

        b1_rep = b1.get_reputation()
        b2_rep = b2.get_reputation()

        game_count = 0
        for game_count in range(max_rounds):
            # b1_move = b1.predict_move(move1_history, score1_history, l_epsilon = epsilon)
            # b2_move = b2.predict_move(move2_history, score2_history, l_epsilon = epsilon)

            b1_move = b1.predict_move(move1_history, move2_history, b2_rep)
            b2_move = b2.predict_move(move2_history, move1_history, b1_rep)

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
            b1.give_feedback(i5, i6, i1, fs1, b2_rep)
            b2.give_feedback(i6, i5, i2, fs2, b1_rep)

        # print(move1_history)
        # print(score1_history)
        # print(move2_history)
        # print(score2_history)

        if learning:
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

        # print('total games', game_count, 'scores:', b1_score_final, b2_score_final, 'bots:',b1.get_id(),b2.get_id(),'elos:', b1.elo, b2.elo)

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
    g = Game(b, n_bot, learning=False, max_rounds=500)
    return analyze_character(g.move1_history, g.move2_history)


def rate_bots_comparative(bots, gen_id):
    res_array = np.zeros((len(bots), len(bots)))

    for i in  range(len(bots)):
        for j in range(len(bots)):
            res_array[i, j] = np.nan

    df = pd.DataFrame(data = res_array,
                      index = [i.get_id() for i in bots],
                      columns=[i.get_id() for i in bots])
    ratings = []

    comp_num = int(max(len(bots) * rating_prob, 2))

    for count1, b1 in enumerate(bots):
        average = 0
        comp_bots = [b for b in bots if b.b_id != b1.b_id]
        comp_bots = random.sample(comp_bots, comp_num)

        for count2, b2 in enumerate(comp_bots):
            if b1.b_id != b2.b_id:
                print(b1.b_id, b2.b_id)
                g1 = Game(b1, b2, learning=False)
                df.loc[b1.get_id(), b2.get_id()] = g1.score[0]
                df.loc[b2.get_id(), b1.get_id()] = g1.score[1]
                # res_array[b2.b_id, b1.b_id] = g1.score[1]
                average += g1.score[0]
        average = average / comp_num
        b1.score = average
        ratings.append({'bot':b1, 'average':average, 'score':average})

    df['average'] = df[[i.get_id() for i in bots]].mean(axis = 1)
    df['stddev'] = df[[i.get_id() for i in bots]].std(axis = 1)
    df['median'] = df[[i.get_id() for i in bots]].median(axis = 1)

    df['elo'] = 0
    df['trainable'] = 0
    df['name'] = 0

    for i in bots:
        df.loc[i.get_id(), 'name'] = i.get_id()
        df.loc[i.get_id(), 'elo'] = i.elo
        df.loc[i.get_id(), 'generation'] = i.generation
        df.loc[i.get_id(), 'depth'] = i.model_depth

        if i.trainable:
            df.loc[i.get_id(), 'trainable'] = 1
        else:
            df.loc[i.get_id(), 'trainable'] = 0

    df['model_depth'] = 0
    df['using_reputation'] = 0
    df['base_alg_random_prob'] = 0

    for i in eff_base_algs:
        for j in ['trainable', 'not_trainable']:
            df[i+ '_' + j] = 0

    for i in bots:
        df.loc[i.get_id(), 'model_depth'] = i.model_depth
        df.loc[i.get_id(), 'using_reputation'] = i.use_reputation
        df.loc[i.get_id(), 'base_alg_random_prob'] = i.base_alg_random_prob

        for j in eff_base_algs:
            if i.alt_base_alg == j and i.trainable:
                df.loc[i.get_id(), j+ '_' + 'trainable'] = 1
                df.loc[i.get_id(), j+ '_' + 'not_trainable'] = 0
            elif i.alt_base_alg == j:
                df.loc[i.get_id(), j+ '_' + 'trainable'] = 0
                df.loc[i.get_id(), j+ '_' + 'not_trainable'] = 1

    df['uncalled_aggression'] = 0
    df['retaliation'] = 0
    df['forgiveness'] = 0
    df['first_move'] = 0

    df['gen_id']= gen_id

    for i in bots:
        char_res = get_character(i)

        i.uncalled_aggression = char_res[0]
        i.retaliation = char_res[1]
        i.forgiveness = char_res[2]
        i.first_move = char_res[3]

        df.loc[i.get_id(), 'uncalled_aggression'] = char_res[0]
        df.loc[i.get_id(), 'retaliation'] = char_res[1]
        df.loc[i.get_id(), 'forgiveness'] = char_res[2]
        df.loc[i.get_id(), 'first_move'] = char_res[3]

    df = df.fillna(df.mean())
    return df, ratings


# for g_count in range(16,17):
#     # bots = [DBot(i, history_len=2, model_depth=8) for i in range(g_count)]


def get_random_new_bot(b_count, generation, history_len=50):
    base_alg = random.choice(base_algorithms)
    # use_reputation = random.randint(0, 1)
    use_reputation = 0
    model_depth = random.randint(min_model_depth, max_model_depth)

    if random.random() < prob_of_trainable:
        trainable = True
    else:
        trainable = False
    alg_constants = (random.randint(1, 3), random.randint(1, 3))
    return DBot(b_count, history_len=history_len, model_depth=model_depth, base_alg=base_alg, trainable=trainable,
         use_reputation=use_reputation, generation=generation, base_alg_random_prob=base_alg_random_prob,
                alg_constants=alg_constants)

max_count = max_num_of_bots
bots = []

for i in range(max_count):
    bots.append(get_random_new_bot(i, 0, history_len=50))
g_count = len(bots)

for b in bots:
    print(b.get_id())

print(len(bots))
# scores = []
score_list = []

gen_data = []

full_results = []

for gen_id in range(generations):
    gc.collect()

    for i in range(100000):
        random.shuffle(bots)
        b1 = bots[0]
        b2 = bots[1]
        print(i, epsilon, b1.b_id, b2.b_id)

        g = Game(b1, b2)
        # scores.append({'s_{0}_score'.format(b1.b_id):g.score[0],
        #                # 's_{0}_history_len'.format(b1.b_id):b1.history_len,
        #                # 's_{0}_history_len'.format(b2.b_id):b1.history_len, 's_{0}_model_depth'.format(b1.b_id): b1.history_len,
        #                # 's_{0}_model_depth'.format(b2.b_id): b1.history_len,
        #                's_{0}_score'.format(b2.b_id):g.score[1]})

        # if b1.b_id > 4 and b2.b_id > 4:
        score_list.append(g.score[0])
        score_list.append(g.score[1])

        if len(score_list) > 0:
            print('trailing results', gen_id, sum(score_list)/len(score_list))
            score_list = score_list[-generation_training_size * 2:]

        if (i%generation_training_size == 0 and i > 0):
            for b in bots:
                if not b.trained_max:
                    b.train_model(always = True, final=True)
                    b.trained_max = True

        num_of_finished_bots = len([b for b in bots if b.trained_max])

        if num_of_finished_bots == max_num_of_bots:
            for b in bots:
                b.trained_max = True

            comp_df, ratings = rate_bots_comparative(bots, gen_id)
            ratings.sort(key = lambda x: x['average'], reverse = True)

            survivors = []
            print('generation : {0}'.format(gen_id))

            for count, i in enumerate([b for b in ratings]):
                if (len(ratings) - count)/len(ratings) > (random.random()*survival_chance_limit):
                    print('selecting bot: {0}, {1}, {2}'.format(i['bot'].get_id(), i['score'], i['bot'].elo))
                else:
                    print('rejecting bot: {0}, {1}, {2}'.format(i['bot'].get_id(), i['score'], i['bot'].elo))

            # for i in [b for b in ratings][:int(len(ratings)*top_survival_chunk_size)]:
            #     if random.random() < top_survival_chunk_rate:
            #         print('selecting bot: {0}, {1}, {2}'.format(i['bot'].get_id(), i['score'], i['bot'].elo))
            #         survivors.append(i['bot'])
            #     else:
            #         print('rejecting bot: {0}, {1}, {2}'.format(i['bot'].get_id(), i['score'], i['bot'].elo))
            #
            # for i in [b for b in ratings][int(len(ratings)*top_survival_chunk_size):]:
            #     if random.random() < bot_survival_chunk_rate:
            #         print('selecting bot: {0}, {1}, {2}'.format(i['bot'].get_id(), i['score'], i['bot'].elo))
            #         survivors.append(i['bot'])
            #     else:
            #         print('rejecting bot: {0}, {1}, {2}'.format(i['bot'].get_id(), i['score'], i['bot'].elo))

            del bots
            gc.collect()
            bots = survivors

            # comp_df, ratings = rate_bots_comparative(bots, gen_id)

            full_results.append(comp_df)
            new_gen_data = {'gen_id': gen_id,
             'average': comp_df['average'].mean(),
             'uncalled_aggression': comp_df['uncalled_aggression'].mean(),
             'retaliation': comp_df['retaliation'].mean(),
             'forgiveness': comp_df['forgiveness'].mean(),
             'first_move': comp_df['first_move'].mean(),
             'depth': comp_df['depth'].mean(),
             'avg_generation': comp_df['generation'].mean(),
             'using_reputation': comp_df['using_reputation'].mean(),
             'trainable': comp_df['trainable'].mean()}

            for b_a in eff_base_algs:
                for t in ['trainable', 'not_trainable']:
                    new_gen_data.update({'strat_' + b_a + '_' + t: comp_df[b_a + '_' + t].mean()})
            gen_data.append(new_gen_data)

            gen_df = pd.DataFrame.from_dict(gen_data)
            gen_df.to_csv(sum_path + 'gen.csv', index = False)

            full_results_df = pd.concat(full_results)
            full_results_df.to_csv(sum_path + 'full_results.csv', index=False)

            print(gen_id)
            comp_df.to_csv(sum_path + 'comp_res_{0}.csv'.format(gen_id))

            while len(bots) < max_num_of_bots:
                max_count += 1
                new_bot =  get_random_new_bot(max_count, gen_id + 1, history_len=50)
                bots.append(new_bot)
            break



# comp_df = rate_bots_comparative(bots)
# comp_df.to_csv('comp_res_final.csv')
# for i in bots:
#     print(i.b_id, rate_Bot(i))

