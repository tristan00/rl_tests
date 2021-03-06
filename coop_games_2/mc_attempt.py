import random
import numpy as np
from sklearn.model_selection import train_test_split
import copy
import pandas as pd
import pickle
import gc
from xgboost import plot_tree, to_graphviz
import xgboost as xgb

elo_d = 500
elo_k = elo_d/10
chance_to_stop = .01
max_iter =  100
forward_view = 10
base_retraining_frequency = .2
generations = 1000
decay_period = 1000
generation_training_size = 100000
max_training_size = 10000000
min_data_to_train = 5
max_num_of_bots = 100
path = r'C:\Users\trist\Documents\prisoner_models\saved_games/'
sum_path =r'C:\Users\trist\Documents\prisoner_models\saved_data/'
starting_epsilon = .1
epsilon = .001
epsilon_decay = .1
epsilon_decay_period = 4
starting_epsilon_decay_period = 50
min_epsilon = .001
maximum_elo = 10000
minimum_elo = 10
starting_elo = 1000
prob_of_trainable = .9
rating_prob = .5
max_tit = 3
max_tat = 3
base_alg_random_prob = 0.01
random_defect_chance = .125
max_model_depth = 12
min_model_depth = 5
history_len = 12

mc_discount_rate = 0
nan_move = np.nan
survival_rate = .8
drop_num = 1 #negative to ignore
randomize_survival_rate = True


# survival_chance_limit = .5 #Scales survival rate of generaltion so scaling survival percentage only starts for the bottom n%, doing it by chance encourages diversity
# min_survival_chance = .5
# max_survival_chance = 1.0
base_algorithms =  ['tit_for_tat', 'random', 'defect', 'no_forgiveness', 'tit_for_2tat', 'cooperate']
base_algorithms =  ['{0}tit_for_{1}tat', 'random', 'defect', 'cooperate', 'no_forgiveness', 'tit_for_tat_first_defect',
                    'tit_for_tat_random_defect', 'tit_for_tat_random_defect', 'tester']
eff_base_algs = ['{0}tit_for_{1}tat'.format(i, j) for i in range(1, max_tit + 1)
                 for j in range(1, max_tat + 1) ] + ['random', 'defect', 'cooperate', 'no_forgiveness',
                                                     'tit_for_tat_first_defect', 'tit_for_tat_random_defect', 'tester']

# base_algorithms = ['random']
# eff_base_algs = ['random']

# base_algorithms = ['random']
# eff_base_algs = ['random']
max_data_size = max_training_size

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


class MC_Model():
    # each input has 2 states, one for each output

    def __init__(self):
        self.data = dict()


    def predict(self, input_data, learning=True):
        if str(input_data.tolist()) in self.data and\
                0 in self.data[str(input_data.tolist())] \
                and 1 in self.data[str(input_data.tolist())]:

            # print('self.data keys', len(self.data))
            s0 = self.data[str(input_data.tolist())].get(0, dict())
            v0 = s0.get('sum', 0)/max(s0.get('count', 0), 1)
            # v0_d = v0*((1-mc_discount_rate)**s0.get('count', 0))
            # v0_d = v0

            s1 = self.data[str(input_data.tolist())].get(1, dict())
            v1 = s1.get('sum', 0)/max(s1.get('count', 0), 1)
            # v1_d = v1 * ((1 - mc_discount_rate) ** s1.get('count', 0))
            # v1_d = v1

            if learning:
                v0_d = v0 * ((1 - mc_discount_rate) ** s0.get('count', 0))
                v1_d = v1 * ((1 - mc_discount_rate) ** s1.get('count', 0))
            else:
                v0_d = v0
                v1_d = v1

            if s0['count'] == 0 and s1['count'] == 0:
                return random.randint(0,1)
            if s0['count'] == 0:
                return 0
            if s1['count'] == 0:
                return 1

            if v1_d > v0_d:
                return 1
            elif v0_d < v1_d:
                return 0



    def take_input(self, input_data, choice, result):
        if str(input_data.tolist()) not in self.data:
            self.data[str(input_data.tolist())] = dict()
        if int(choice) not in self.data[str(input_data.tolist())]:
            self.data[str(input_data.tolist())][int(choice)] = {'sum':0, 'count':0}
        # self.data[str(input_data.tolist())][int(choice)].setdefault({'sum':0, 'count':0})
        self.data[str(input_data.tolist())][int(choice)]['sum'] += result
        self.data[str(input_data.tolist())][int(choice)]['count'] += 1


xgb_params = {
    'learning_rate':.1,
    }


def calculate_new_elo(outcome, player_1_elo, player_2_elo):
    expected_outcome = 1 / (1 + 10**((player_2_elo - player_1_elo)/elo_d))
    new_elo =  player_1_elo + (elo_k * (outcome-expected_outcome))
    return min(maximum_elo, max(minimum_elo, new_elo))


class DBot():
    def __init__(self, n, model_depth = 1, base_alg = 'random', trainable = False, generation = 0, use_reputation = 0, base_alg_random_prob = 0, alg_constants = (1, 1), base_learner_type = 'gbm'):
        self.base_learner_type = base_learner_type
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
        self.decay_period = starting_epsilon_decay_period
        self.epsilon = starting_epsilon
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
            ['opponent_move_{0}_moves_past'.format(i) for i in range(self.history_len, 0, -1)] + ['next_move', 'game_current_len']

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
        # for count, tree_in_forest in enumerate(self.model.estimators_):
        #     export_graphviz(tree_in_forest, out_file=path + '/agent_{0}_{1}.txt'.format(self.get_id(), count),
        #                         feature_names=self.feature_names)
        # export_graphviz(self.model, out_file=path + '/agent_{0}.txt'.format(self.get_id()), feature_names=self.feature_names)

        if self.base_learner_type == 'gbm' and False:
            a = to_graphviz(self.model, num_trees=self.model.best_iteration)
            with open(path + '/agent_{0}.txt'.format(self.get_id()), 'w') as f:
                f.write(str(a))


    def get_id(self):
        if self.trainable or self.use_model:

            if self.base_learner_type == 'gbm':
                model_str = 'trainable' + '_model_depth_' + str(self.model_depth)
            else:
                model_str = 'trainable_mc_'

            if self.use_reputation:
                return str(self.b_id) + '_' + 'generation_' + str(self.generation) + '_' + self.alt_base_alg + '_' + model_str + '_using_reputation'
            else:
                return str(self.b_id) + '_' + 'generation_' + str(self.generation) + '_' + self.alt_base_alg + '_' + model_str + '_not_using_reputation'

        else:
            return str(self.b_id) + '_' + self.alt_base_alg + '_' + 'not_trainable'


    def train_model(self, always = False, final = False):

        training_prob = max(min_epsilon, self.base_training_rate * (.5 ** (self.g_count / self.decay_period)))
        # print(len(self.x), training_prob, self.trained_max)
        # print('training_prob', training_prob)


        if (len(self.x) > 0 and len(self.x) > min_data_to_train and self.trainable and (random.random() < training_prob or always) and not self.trained_max)\
                or (not self.trained_max and len(self.x) >= max_data_size):
            print('training model', training_prob)
            x = np.array(self.x)
            y = np.array(self.y)

            if x.shape[0] > max_training_size:
                print('agent {0} sampling'.format(self.b_id))
                sample_portion = 1- max_training_size/x.shape[0]
                x, _, y, _ = train_test_split(x, y, test_size = sample_portion)

            # # self.model = RandomForestRegressor(max_depth=self.model_depth, n_estimators=10)
            # self.model = DecisionTreeRegressor(max_depth=self.model_depth)
            # # self.model = RandomForestRegressor(max_depth=self.model_depth, n_estimators=10)

            # self.model = XGBRegressor()

            if self.base_learner_type == 'gbm':
                x, x2, y, y2 = train_test_split(x, y, test_size=.1)
                dtrain = xgb.DMatrix(x, label = y, feature_names=self.feature_names)
                dval = xgb.DMatrix(x2, label = y2, feature_names=self.feature_names)
                watchlist = [(dtrain, 'train'), (dval, 'eval')]
                xgb_params_temp = copy.deepcopy(xgb_params)
                xgb_params_temp['max_depth'] = self.model_depth
                self.model = xgb.train(xgb_params_temp, dtrain, 10, watchlist, early_stopping_rounds=2)
            # else:
            #     self.model = MC_Model()
            #     nm_col = self.feature_names.index('next_move')
            #
            #     # x2 = x[:,nm_col]
            #     # x1 = np.hstack([x[:nm_col,nm_col], x[nm_col+1:,nm_col]])
            #
            #     x1 =np.delete(x, nm_col, axis = 1)
            #     x1 = np.delete(x1, x1.shape[1] - 1, axis=1)
            #     x2 = x[:,nm_col]
            #     # x1 = np.hstack([x[nm_col, :nm_col], x[nm_col + 1:, nm_col]])
            #     for x_i, x_m, v in zip(x1, x2, y):
            #         self.model.take_input(x_i, x_m, v)

            self.use_model = True
            self.save_model()

            if not self.trained_max and (len(self.x) >= max_data_size or final):
                print(self.get_id(), 'model training done', len(self.x))
                self.trained_max = True


    def give_feedback(self,  move_history1, move_history2, move, score, opponent_reputation):
        if len(self.x) < max_data_size and not self.trained_max:
            small_move1_history = move_history1[-self.history_len:]
            small_move2_history = move_history2[-self.history_len:]

            game_len = len(move_history1)

            while len(small_move1_history) < self.history_len:
                small_move1_history.insert(0, nan_move)
            while len(small_move2_history) < self.history_len:
                small_move2_history.insert(0, nan_move)

            if self.use_reputation:
                new_x = small_move1_history + small_move2_history + [move] + opponent_reputation + [game_len]
            else:
                new_x = small_move1_history + small_move2_history + [move] + [game_len]

            if self.base_learner_type != 'gbm' and self.trainable and not self.model:
                self.model = MC_Model()
            if self.base_learner_type != 'gbm' and self.trainable and not self.trained_max:
                x0_m = np.array([new_x])
                nm_col = self.feature_names.index('next_move')
                x_d = np.delete(x0_m, nm_col, axis=1)
                x_d = np.delete(x_d, x_d.shape[1] - 1, axis=1)
                self.model.take_input(x_d, move, score)

            self.x.append(new_x)
            self.y.append(score)



    def predict_move(self, b1_move_history, b2_move_history, rep, opponent_id = 0, move_num = 0, l_epsilon = 1.0, learning=True):
        # print(move_history, score_history)

        epsilon = self.epsilon * ((1 - epsilon_decay)**(self.g_count / self.decay_period))

        # if random.random() < .01:
        #     print('epsilon', self.epsilon, epsilon, epsilon_decay, self.g_count, self.decay_period)

        if self.use_model and random.random() > epsilon and self.model:
            small_move_history = b1_move_history[-self.history_len:]
            small_score_history = b2_move_history[-self.history_len:]
            game_len = len(b1_move_history)

            while len(small_move_history) < self.history_len:
                small_move_history.insert(0, nan_move)
            while len(small_score_history) < self.history_len:
                small_score_history.insert(0, nan_move)

            if self.use_reputation:
                x1 = small_move_history + small_score_history + [1] + rep + [game_len]
                x0 = small_move_history + small_score_history + [0] + rep + [game_len]
            else:
                x1 = small_move_history + small_score_history + [1] + [game_len]
                x0 = small_move_history + small_score_history + [0] + [game_len]

            if self.base_learner_type == 'gbm':
                # dnn_input1 = self.scaler.transform(np.array([x1]).astype(np.float32))
                # dnn_input0 = self.scaler.transform(np.array([x0]).astype(np.float32))
                #
                # x1_val = self.dnn.predict(dnn_input1)
                # x0_val = self.dnn.predict(dnn_input0)
                x1_m = xgb.DMatrix(np.array([x1]), feature_names=self.feature_names)
                x0_m = xgb.DMatrix(np.array([x0]), feature_names=self.feature_names)

                x1_val = self.model.predict(x1_m)
                x0_val = self.model.predict(x0_m)

                # print(x0_val[0], x1_val[0])

                if x1_val[0] > x0_val[0]:
                    return 1
                elif x1_val[0] < x0_val[0]:
                    return 0
            else:
                x0_m = np.array([x0])

                nm_col = self.feature_names.index('next_move')

                x_d = np.delete(x0_m, nm_col, axis=1)
                x_d = np.delete(x_d, x_d.shape[1] - 1, axis=1)
                # x_m = x0_m[:, nm_col]
                # x_d = np.hstack([x0_m[:nm_col, nm_col], x0_m[nm_col + 1:, nm_col]])
                pred_move =  self.model.predict(x_d, learning=learning)
                if pred_move:
                    return pred_move
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

            b1_move = b1.predict_move(move1_history, move2_history, b2_rep, learning=learning)
            b2_move = b2.predict_move(move2_history, move1_history, b1_rep, learning=learning)

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

            fs1 = sum(score1_history[count:])/len(score1_history[count:])
            fs2 = sum(score2_history[count:])/len(score2_history[count:])
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
        # print(game_count)
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
    n_bot = DBot(1, model_depth=1, base_alg='preset', trainable=False)
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
        # try:
        #     comp_bots = random.sample(comp_bots, comp_num)
        # except:
        #     pass

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

        if i.base_learner_type != 'gbm':
            df.loc[i.get_id(), 'mc'] = 1
        else:
            df.loc[i.get_id(), 'mc'] = 0

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


def get_random_new_bot(b_count, generation):
    base_alg = random.choice(base_algorithms)
    # use_reputation = random.randint(0, 1)
    use_reputation = 0
    model_depth = random.randint(min_model_depth, max_model_depth)

    if random.random() < prob_of_trainable:
        trainable = True
    else:
        trainable = False
    # possible_base_learner_types = ['gbm', 'mc']
    possible_base_learner_types = ['mc']
    alg_constants = (random.randint(1, 3), random.randint(1, 3))
    return DBot(b_count, model_depth=model_depth, base_alg=base_alg, trainable=trainable,
         use_reputation=use_reputation, generation=generation, base_alg_random_prob=base_alg_random_prob,
                alg_constants=alg_constants, base_learner_type=random.choice(possible_base_learner_types))

max_count = max_num_of_bots
bots = []

for i in range(max_count):
    bots.append(get_random_new_bot(i, 0))
g_count = len(bots)

for b in bots:
    print(b.get_id())

print(len(bots))
# scores = []
score_list = []

gen_data = []

full_results = []


result_history = []
for gen_id in range(generations):
    gc.collect()
    score_list = []
    for i in range(10000000):
        random.shuffle(bots)
        b1 = bots[0]
        b2 = bots[1]
        print(i, epsilon, b1.b_id, b2.b_id)

        if not(b1.trainable and b2.trainable and b1.trained_max and b2.trained_max):
            g = Game(b1, b2)
            score_list.append(g.score[0])
            score_list.append(g.score[1])

        if len(score_list) > 0:
            print('trailing results', gen_id, sum(score_list)/len(score_list))
            # score_list = score_list[-2000:]

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
            # for count, i in enumerate([b for b in ratings]):
            #
            #     scaled_rank = (min_survival_chance + ((len(ratings) - count)/len(ratings))*(max_survival_chance - min_survival_chance))
            #     print(count, scaled_rank)
            #     if scaled_rank > random.random():
            #         print('selecting bot: {0}, {1}, {2}'.format(i['bot'].get_id(), i['score'], i['bot'].elo))
            #         survivors.append(i['bot'])
            #     else:
            #         print('rejecting bot: {0}, {1}, {2}'.format(i['bot'].get_id(), i['score'], i['bot'].elo))
            # print(len(survivors))
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

            if drop_num > 0:
                survivor_index = len(bots) - drop_num
            else:
                survivor_index = int(len(ratings) * (1 - ((1 - survival_rate) * random.random())))

            result_history.append(comp_df['average'].mean())
            print('generation : {0}'.format(gen_id))
            for i in [b for b in ratings][:survivor_index]:
                print('selecting bot: {0}, {1}, {2}'.format(i['bot'].get_id(), i['score'], i['bot'].elo))
            for i in [b for b in ratings][survivor_index:]:
                print('rejecting bot: {0}, {1}, {2}'.format(i['bot'].get_id(), i['score'], i['bot'].elo))
            print(result_history)
            del bots
            gc.collect()
            bots = [b['bot'] for b in ratings][:survivor_index]

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
             'trainable': comp_df['trainable'].mean(),
             'mc': comp_df['mc'].mean()}

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
                new_bot =  get_random_new_bot(max_count, gen_id + 1)
                bots.append(new_bot)
            break



