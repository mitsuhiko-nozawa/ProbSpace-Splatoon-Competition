from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Concatenate, Embedding, Activation
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd

def make_model(df1, df2, categorical_feature):
    all_df = pd.concat([df1, df2])
    numeric_feature_num = int(df1.columns .shape[0]) - len(categorical_feature)
    #numeric_feature_num = int((df.dtypes == float).sum() + (df.dtypes == int).sum())
    #print(numeric_feature_num)
    num_input = Input(numeric_feature_num, )
    inputs = []
    embs = []
    embedding_dim = 6
    for col in categorical_feature:
        value_size = all_df[col].unique().shape[0]
        input_cat = Input(1,)
        #emb = Embedding(input_dim=value_size, output_dim=embedding_dim)(input_cat)
        emb = Embedding(input_dim=value_size, output_dim=int(np.log2(value_size))+1)(input_cat)
        emb = Flatten()(emb)
        #emb = Dropout(0.2)(emb)
        #emb = Dense(2, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(emb)
        #emb = Flatten()(emb)

        inputs.append(input_cat)
        embs.append(emb)

    concat = Concatenate()([num_input, *embs])

    outputs = (Dense(1024, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))(concat)
    outputs = (Activation('relu'))(outputs)
    outputs = (Dropout(.4))(outputs)
    #outputs = (Dense(256))(outputs)
    #outputs = (Activation('relu'))(outputs)
    #outputs = (Dropout(.4))(outputs)
    outputs = (Dense(128, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))(outputs)
    outputs = (Activation('relu'))(outputs)
    outputs = (Dropout(.4))(outputs)
    outputs = (Dense(1))(outputs)
    outputs = (Activation('sigmoid'))(outputs)
    model = Model(inputs=[num_input, *inputs], outputs=[outputs])
    model.compile(optimizer=Adam(lr=0.0001), loss="binary_crossentropy")
    #display(model.summary())

    return model

def prepro_nn(df, categorical_col):
    data = []
    numeric_cols = [col for col in df.columns if col not in categorical_col]
    data.append(df[numeric_cols].values)
    for col in categorical_col:
        data.append(np.array(df[col].values).reshape(-1, 1))
    return data
