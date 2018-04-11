import numpy as np
import pandas as pd

import os
import time

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from neuralnet import NeuralNetwork


def clean_inputs(x):
    # Replace null values with most frequent value
    x.loc[x['Sex'].isnull(), 'Sex'] = x['Sex'].value_counts().idxmax()
    x.loc[x['Embarked'].isnull(), 'Embarked'] = x['Embarked'].value_counts().idxmax()
    x.loc[x['Age'].isnull(), 'Age'] = x['Age'].value_counts().idxmax()

    # Convert strings to int-categories
    label_encoder = LabelEncoder()
    sex_int_encoded = label_encoder.fit_transform(x.loc[:, 'Sex'].values)

    # Convert int-categories to binary-categories
    sex_int_encoded = sex_int_encoded.reshape(len(sex_int_encoded), 1)
    sex_one_hot_encoded = OneHotEncoder(sparse=False).fit_transform(sex_int_encoded)

    # Inverting back to our orginal label
    # inverted = label_encoder.inverse_transform([np.argmax(sex_one_hot_encoded[0, :])])
    x['Female'] = np.NaN
    x['Male'] = np.NaN
    x.loc[:, ['Female', 'Male']] = sex_one_hot_encoded
    x.drop(['Sex', 'Embarked'], inplace=True, axis=1)
    return x


def clean_outputs(y):
    doa = y.values
    doa = doa.reshape(len(doa), 1)
    doa_one_hot_encoded = OneHotEncoder(sparse=False).fit_transform(doa)
    y['Dead'] = np.NaN
    y['Survived'] = np.NaN
    y[['Dead', 'Survived']] = doa_one_hot_encoded
    return y


def prep_train_data():
    # Load and select feature cols
    df = pd.read_csv('./data/train.csv', index_col=0)
    x = df.loc[:, ['Pclass', 'Sex', 'Age', 'Embarked']]
    y = df.loc[:, ['Survived']]
    x = clean_inputs(x)
    y = clean_outputs(y)
    return x, y


def prep_test_data():
    # Load and select feature cols
    df = pd.read_csv('./data/test.csv', index_col=0)
    x = df.loc[:, ['Pclass', 'Sex', 'Age', 'Embarked']]
    y = pd.read_csv('./data/gender_submission.csv', index_col=0)
    x = clean_inputs(x)
    y = clean_outputs(y)
    return x, y


def categorize_output(raw_outputs, test_y):
    test_y['pred_survived'] = np.NaN
    test_y['pred_dead'] = np.NaN
    test_y[['pred_survived', 'pred_dead']] = raw_outputs
    test_y['pred_survived'] = [np.rint(x) for x in test_y['pred_survived']]
    test_y['pred_dead'] = [np.rint(x) for x in test_y['pred_dead']]
    cost = ((test_y['pred_survived'] - test_y['Survived']) ** 2).mean()
    test_y = test_y[['pred_survived']]
    test_y.columns = ['Survived']
    return test_y, cost


if __name__ == '__main__':
    x, y = prep_train_data()
    test_x, test_y = prep_test_data()
    h_layers = [10, 10]
    inputs = x.values
    outputs = y.values
    nn = NeuralNetwork(n_inputs=4,
                       n_outputs=2,
                       h_layers=h_layers,
                       inputs=inputs,
                       expected_outputs=outputs)
    nn.open_session()
    nn.make_model(is_weights_rand=True)
    nn.make_tensor_board()
    s = time.time()
    nn.train(epochs=999, learning_rate=0.01, isliveplot=False)
    raw_outputs, _ = nn.test(test_inputs=test_x.values, test_outputs=test_y.values)
    test_y, cost = categorize_output(raw_outputs, test_y)
    test_y.to_csv('./data/my_submission.csv')
    print('cost: ', cost)
    e = time.time() - s
    print("Training took ", e, "seconds")
    nn.close_session()
    nn.plot()