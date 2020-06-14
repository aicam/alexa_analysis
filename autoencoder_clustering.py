import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix,classification_report
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans


df = pd.read_csv('dataset/alexa.com_site_info.csv')

df = df.dropna()

cols3 = ['all_topics_easy_to_rank_keywords_search_pop_parameter_1',
       'all_topics_easy_to_rank_keywords_search_pop_parameter_2',
       'all_topics_easy_to_rank_keywords_search_pop_parameter_3',
       'all_topics_easy_to_rank_keywords_search_pop_parameter_4']

cols2 = ['all_topics_easy_to_rank_keywords_relevance_to_site_parameter_1',
       'all_topics_easy_to_rank_keywords_relevance_to_site_parameter_2',
       'all_topics_easy_to_rank_keywords_relevance_to_site_parameter_3',
       'all_topics_easy_to_rank_keywords_relevance_to_site_parameter_4',]

cols4 = ['all_topics_keyword_gaps_Avg_traffic_parameter_1',
       'all_topics_keyword_gaps_Avg_traffic_parameter_2',
       'all_topics_keyword_gaps_Avg_traffic_parameter_3',
       'all_topics_keyword_gaps_Avg_traffic_parameter_4']

##########################################################################################

cols5 = ['all_topics_keyword_gaps_search_popularity_parameter_1',
       'all_topics_keyword_gaps_search_popularity_parameter_2',
       'all_topics_keyword_gaps_search_popularity_parameter_3',
       'all_topics_keyword_gaps_search_popularity_parameter_4']

cols6 = ['all_topics_buyer_keywords_Avg_traffic_parameter_1',
        'all_topics_buyer_keywords_Avg_traffic_parameter_2',
        'all_topics_buyer_keywords_Avg_traffic_parameter_3',
        'all_topics_buyer_keywords_Avg_traffic_parameter_4']

cols7 = ['all_topics_buyer_keywords_organic_competition_parameter_1',
        'all_topics_buyer_keywords_organic_competition_parameter_2',
        'all_topics_buyer_keywords_organic_competition_parameter_3',
        'all_topics_buyer_keywords_organic_competition_parameter_4']

cols8 = ['all_topics_optimization_opportunities_search_pop_parameter_1',
        'all_topics_optimization_opportunities_search_pop_parameter_2',
        'all_topics_optimization_opportunities_search_pop_parameter_3',
        'all_topics_optimization_opportunities_search_pop_parameter_4']

cols9 = ['all_topics_optimization_opportunities_organic_share_of_voice_parameter_1',
        'all_topics_optimization_opportunities_organic_share_of_voice_parameter_2',
        'all_topics_optimization_opportunities_organic_share_of_voice_parameter_3',
        'all_topics_optimization_opportunities_organic_share_of_voice_parameter_4']

cols10 = ['all_topics_top_keywords_search_traffic_parameter_1',
         'all_topics_top_keywords_search_traffic_parameter_2',
         'all_topics_top_keywords_search_traffic_parameter_3',
         'all_topics_top_keywords_search_traffic_parameter_4']

cols11 = ['all_topics_top_keywords_share_of_voice_parameter_1_percentage',
         'all_topics_top_keywords_share_of_voice_parameter_2_percentage',
         'all_topics_top_keywords_share_of_voice_parameter_3_percentage',
         'all_topics_top_keywords_share_of_voice_parameter_4_percentage']

cols12 = ['audience_overlap_sites_overlap_scores_parameter_1',
         'audience_overlap_sites_overlap_scores_parameter_2',
         'audience_overlap_sites_overlap_scores_parameter_3',
         'audience_overlap_sites_overlap_scores_parameter_4',
         'audience_overlap_sites_overlap_scores_parameter_5']

cols13 = ['audience_overlap_similar_sites_to_this_site_parameter_1',
         'audience_overlap_similar_sites_to_this_site_parameter_2',
         'audience_overlap_similar_sites_to_this_site_parameter_3',
         'audience_overlap_similar_sites_to_this_site_parameter_4',
         'audience_overlap_similar_sites_to_this_site_parameter_5']

df.loc[:,'all_topics_easy_to_rank_relevance_to_site_average'] = df.loc[:, cols2].mean(axis=1)
df.loc[:,'all_topics_easy_to_rank_keywords_search_pop_average'] = df.loc[:, cols3].mean(axis=1)
df.loc[:,'all_topics_keyword_gaps_Avg_traffic_average'] = df.loc[:, cols4].mean(axis=1)
df.loc[:,'all_topics_keyword_gaps_search_popularity_average'] = df.loc[:, cols5].mean(axis=1)

df.loc[:,'all_topics_buyer_keywords_Avg_traffic_average'] = df.loc[:, cols6].mean(axis=1)
df.loc[:,'all_topics_buyer_keywords_organic_competition_average'] = df.loc[:, cols7].mean(axis=1)
df.loc[:,'all_topics_optimization_opportunities_search_pop_average'] = df.loc[:, cols8].mean(axis=1)
df.loc[:,'all_topics_optimization_opportunities_organic_share_of_voice_average'] = df.loc[:, cols9].mean(axis=1)
df.loc[:,'all_topics_top_keywords_search_traffic_average'] = df.loc[:, cols10].mean(axis=1)
df.loc[:,'all_topics_top_keywords_share_of_voice_average'] = df.loc[:, cols11].mean(axis=1)
df.loc[:,'audience_overlap_sites_overlap_scores_average'] = df.loc[:, cols12].mean(axis=1)
df.loc[:,'audience_overlap_similar_sites_to_this_site_average'] = df.loc[:, cols13].mean(axis=1)
import re
p = re.compile('.*_average')
x = df[[s for s in df.columns if p.match(s)]].to_numpy()
x = np.delete(x, 11, axis=1)

# Autoencoders
def autoencoder(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    input_img = Input(shape=(dims[0],), name='input')
    x = input_img
    # internal layers in encoder
    for i in range(n_stacks-1):
        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)

    # hidden layer
    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here

    x = encoded
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

    # output
    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
    decoded = x
    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')
dims = [11, 10, 8, 7]
init = VarianceScaling(scale=1. / 3., mode='fan_in',
                           distribution='uniform')
pretrain_optimizer = SGD(lr=0.1, momentum=0.9)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters= 8, )

# dims represents the dense layer units number : 5 layers have each unit cell number
autoencoder, encoder = autoencoder(dims, init=init)
autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
autoencoder.fit(x, x, epochs=20) #, callbacks=cb)
