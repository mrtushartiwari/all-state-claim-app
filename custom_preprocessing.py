import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import pickle
import xgboost as xgb
import time

SHIFT = 200

continous_features = []
for i in range(1,15):
    continous_features.append('cont' + str(i))
categorical_features = []
for i in range(1,117):
    categorical_features.append('cat' + str(i))


def encode_continous(df,continous_features):
    for col in continous_features:
        df[col +'_log'] = np.log1p(df[col].astype(float) ) # Log transformed
        df[col + '_squareroot'] = np.sqrt(df[col].astype(float))  # Square root
        df[col + '_square'] = np.square(df[col].astype(float))  # Square
        df[col + '_log2'] = np.log2(df[col].astype(float))  # log2


def encode_category(df,categorical_features):
    for col in categorical_features:
        unique_classes = sorted(df[col].unique())
        encoded_class = dict()

        for c in unique_classes:
            if len(c) == 2:
                encoded_class[c] = ord(c[0]) + ord(c[1]) - (2*65)  
                continue 
            encoded_class[c] = ord(c) - 65
        df[col].replace(encoded_class ,inplace = True)



with open("base_models_pickle", "rb") as fp:   # Unpickling
    base_models = pickle.load(fp)

with open("meta_models_pickle", "rb") as fp:   # Unpickling
    meta_models = pickle.load(fp)

def final_fun_1(df):
    '''  Fuction takes the input as a dataframe containing the features and returns the losses of the input claim records'''
    # Preprocessing the inputs
    encode_category(df,categorical_features)
    encode_continous(df,continous_features)
    df.replace([np.inf, -np.inf], -9, inplace=True)

    meta_df =  pd.DataFrame()
    #meta_df['loss'] = np.log(df['loss'].values) + SHIFT

    #df.drop(columns= ['id','loss'], inplace=True)

    num = 1
    for model in base_models:
        meta_df['prediction_' + str(num)] = model.predict(df)
        num += 1

    predictors = []
    for i in meta_df:
        if 'predict' in i:
            predictors.append(i)

    loss_intermediate = meta_models.predict(meta_df[predictors])
    losses = np.exp(loss_intermediate - SHIFT )
    return losses


def final_fun_2(df,target_loss):
    ''' Fuction takes the input as a dataframe containing the features and the target_losss values returns the average MAE  '''
    # Preprocessing the inputs
    encode_category(df,categorical_features)
    encode_continous(df,continous_features)
    df.replace([np.inf, -np.inf], -9, inplace=True)


    meta_df =  pd.DataFrame()
    #meta_df['loss'] = np.log(df['loss'].values) + SHIFT

    #df.drop(columns= ['id','loss'], inplace=True)

    num = 1
    for model in base_models:
        meta_df['prediction_' + str(num)] = model.predict(df)
        num += 1

    predictors = []
    for i in meta_df:
        if 'predict' in i:
            predictors.append(i)
    #print(meta_df)

    loss_intermediate = meta_models.predict(meta_df[predictors])
    losses = np.exp(loss_intermediate - SHIFT )
    return mean_absolute_error(target_loss,losses)