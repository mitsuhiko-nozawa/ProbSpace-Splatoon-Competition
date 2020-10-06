from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Concatenate, Embedding, Activation
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd

def make_model(df, categorical_feature):
    numeric_feature_num = int(df.columns .shape[0]) - len(categorical_feature)
    #numeric_feature_num = int((df.dtypes == float).sum() + (df.dtypes == int).sum())
    #print(numeric_feature_num)
    num_input = Input(numeric_feature_num, )
    inputs = []
    embs = []
    embedding_dim = 10
    for col in categorical_feature:
        value_size = df[col].unique().shape[0]
        input_cat = Input(1,)
        emb = Embedding(input_dim=value_size, output_dim=embedding_dim)(input_cat)
        emb = Flatten()(emb)
        inputs.append(input_cat)
        embs.append(emb)

    concat = Concatenate()([num_input, *embs])

    outputs = (Dense(1024))(concat)
    outputs = (Activation('relu'))(outputs)
    outputs = (Dropout(.35))(outputs)
    outputs = (Dense(256))(outputs)
    outputs = (Activation('relu'))(outputs)
    outputs = (Dropout(.15))(outputs)
    # outputs = (Dense(32))(outputs)
    # outputs = (Activation('relu'))(outputs)
    # outputs = (Dropout(.15))(outputs)
    outputs = (Dense(1))(outputs)
    outputs = (Activation('sigmoid'))(outputs)
    model = Model(inputs=[num_input, *inputs], outputs=[outputs])
    model.compile(optimizer=Adam(lr=0.01), loss="binary_crossentropy")
    return model

def prepro_nn(df, categorical_col):
    data = []
    numeric_cols = [col for col in df.columns if col not in categorical_col]
    data.append(df[numeric_cols].values)
    for col in categorical_col:
        data.append(np.array(df[col].values).reshape(-1, 1))
    return data
