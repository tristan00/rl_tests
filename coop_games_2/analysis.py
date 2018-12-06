import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np



sum_path =r'C:\Users\trist\Documents\prisoner_models\saved_data/'
all_data_df = pd.read_csv(sum_path + 'full_results.csv')
all_data_df['using_reputation'] = all_data_df['using_reputation'].apply(lambda x: 1 if x else 0)

all_data_df = all_data_df.fillna(all_data_df.median())
all_data_df_features = all_data_df[['average', 'depth', 'elo', 'first_move', 'forgiveness', 'median', 'retaliation', 'stddev', 'uncalled_aggression']]

scaler = MinMaxScaler()
all_data_df_features = scaler.fit_transform(all_data_df_features)


kmeans = KMeans(n_clusters=12)
clusters = kmeans.fit_predict(all_data_df_features)

enc = OneHotEncoder(sparse = False)
clusters_encoded = enc.fit_transform(np.reshape(clusters, (-1, 1)))
clusters_df = pd.DataFrame(data=clusters_encoded,
                           columns = ['cluster_{0}'.format(i) for i in range(12)],
                           index = all_data_df.index)

# all_data_df_features = all_data_df_features.join(clusters_df)
all_data_df = all_data_df.join(clusters_df)
print(all_data_df)

