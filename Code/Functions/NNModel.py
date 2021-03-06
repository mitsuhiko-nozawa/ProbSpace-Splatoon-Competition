from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Concatenate, Embedding, Activation
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd



def prepro_nn(df, categorical_col):
    data = []
    numeric_cols = [col for col in df.columns if col not in categorical_col]
    data.append(df[numeric_cols].values)
    for col in categorical_col:
        data.append(np.array(df[col].values).reshape(-1, 1))
    return data


def make_model(df1, df2, categorical_feature):
    all_df = pd.concat([df1, df2])
    numeric_feature_num = int(df1.columns.shape[0]) - len(categorical_feature)
    #numeric_feature_num = int((df.dtypes == float).sum() + (df.dtypes == int).sum())
    #print(numeric_feature_num)
    num_input = Input(numeric_feature_num, )
    inputs = []
    embs = []
    embedding_dim = 6
    for col in categorical_feature:
        value_size = all_df[col].unique().shape[0]
        input_cat = Input(1,)
        emb = Embedding(input_dim=value_size, output_dim=int(np.log2(value_size))+1)(input_cat)
        emb = Flatten()(emb)

        inputs.append(input_cat)
        embs.append(emb)

    concat = Concatenate()([num_input, *embs])

    outputs = Dense(1024)(concat)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(.4)(outputs)

    outputs = Dense(128)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(.4)(outputs)

    outputs = Dense(1)(outputs)
    outputs = Activation('sigmoid')(outputs)

    model = Model(inputs=[num_input, *inputs], outputs=[outputs])
    model.compile(optimizer=Adam(lr=0.0001), loss="binary_crossentropy")

    return model


def make_model2(df1, df2, categorical_feature):
    all_df = pd.concat([df1, df2])
    numeric_feature_num = int(df1.columns.shape[0]) - int(len(categorical_feature))
    print(numeric_feature_num)
    print("sss")

    print(1)
    num_input = Input(numeric_feature_num, )
    inputs = []
    embs = []
    
    for col in categorical_feature:
        value_size = all_df[col].unique().shape[0]
        input_cat = Input(1, )
        emb = Embedding(input_dim=value_size, output_dim=int(np.log2(value_size)) + 1)(input_cat)
        emb = Flatten()(emb)

        inputs.append(input_cat)
        embs.append(emb)

    outputs = Concatenate()([num_input, *embs])
    outputs = Dropout(0.13)(outputs)

    outputs = Dense(2000)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(0.7)(outputs)


    outputs = Dense(1200)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dropout(.4)(outputs)

    outputs = Dense(40)(outputs)
    outputs = Activation('relu')(outputs)

    outputs = Dense(1)(outputs)
    outputs = (Activation('sigmoid'))(outputs)

    model = Model(inputs=[num_input, *inputs], outputs=[outputs])
    model.compile(optimizer=Adam(lr=0.0001), loss="binary_crossentropy")

    return model





def make_model3(df1, df2):

    inputs = Input(df1.shape[1],)
    outputs = Dense(128)(inputs)
    outputs = Activation('relu')(outputs)

    outputs = Dense(32)(outputs)
    outputs = Activation('relu')(outputs)

    outputs = Dense(1)(outputs)
    outputs = (Activation('sigmoid'))(outputs)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(lr=0.0001), loss="binary_crossentropy")

    return model

def make_model4(df1, df2):

    inputs = Input((df1.shape[1],))
    outputs = Dense(32)(inputs)
    outputs = Activation('relu')(outputs)

    outputs = Dense(1)(outputs)
    outputs = (Activation('sigmoid'))(outputs)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(lr=0.001), loss="binary_crossentropy")

    return model