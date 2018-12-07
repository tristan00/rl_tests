import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np
import pickle


sum_path =r'C:\Users\trist\Documents\prisoner_models\saved_data/'
all_data_df = pd.read_csv(sum_path + 'full_results.csv')
all_data_df['using_reputation'] = all_data_df['using_reputation'].apply(lambda x: 1 if x else 0)

all_data_df = all_data_df.fillna(all_data_df.median())
all_data_df_features = all_data_df[['first_move', 'forgiveness', 'retaliation','uncalled_aggression']]

scaler = MinMaxScaler()
all_data_df_features = scaler.fit_transform(all_data_df_features)

try:
    with open(sum_path + 'kmeans.plk', 'rb') as f:
        kmeans = KMeans(n_clusters=12)
        clusters = kmeans.predict(all_data_df_features)
        enc = OneHotEncoder(sparse=False)
        clusters_encoded = enc.fit_transform(np.reshape(clusters, (-1, 1)))
        clusters_df = pd.DataFrame(data=clusters_encoded,
                                   columns=['cluster_{0}'.format(i) for i in range(12)],
                                   index=all_data_df.index)
except:
    with open(sum_path + 'kmeans.plk', 'wb') as f:
        kmeans = KMeans(n_clusters=12)
        clusters = kmeans.fit_predict(all_data_df_features)
        enc = OneHotEncoder(sparse = False)
        clusters_encoded = enc.fit_transform(np.reshape(clusters, (-1, 1)))
        clusters_df = pd.DataFrame(data=clusters_encoded,
                                   columns = ['cluster_{0}'.format(i) for i in range(12)],
                                   index = all_data_df.index)

# all_data_df_features = all_data_df_features.join(clusters_df)
all_data_df = all_data_df.join(clusters_df)
group_features = ['cluster_0', 'cluster_1', 'cluster_2', 'cluster_3', 'cluster_4', 'cluster_5', 'cluster_6', 'cluster_7',
                  'cluster_8', 'cluster_9', 'cluster_10', 'cluster_11', 'using_reputation', 'uncalled_aggression',
                  'trainable', 'stddev', 'retaliation', 'forgiveness', 'elo', 'average', 'median']


cluster_characteristics = all_data_df.groupby(['cluster_{0}'.format(i) for i in range(12)])[group_features].mean()
generation_characteristics = all_data_df.groupby('gen_id')['median'].rank(ascending =False)
all_data_df['generation_rank'] = generation_characteristics

#get only survivors

generation_characteristics_median_df = all_data_df[all_data_df['generation_rank'] <= 80]
# generation_characteristics_median_df = generation_characteristics_median_df.groupby(['gen_id'])[group_features].median()
# generation_characteristics_mean_df = all_data_df[all_data_df['generation_rank'] <= 80]
# generation_characteristics_mean_df = generation_characteristics_mean_df.groupby(['gen_id'])[group_features].mean()
#
# generation_characteristics_median_df.to_csv(sum_path + 'generation_mean.csv')
# generation_characteristics_mean_df.to_csv(sum_path + 'generation_median.csv')
# cluster_characteristics.to_csv(sum_path + 'cluster_characteristics.csv')
#

generation_characteristics_median_df

