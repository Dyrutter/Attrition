import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
import json


def get_df(dataset='finaldata.csv'):
    """
    Load config.json. Get input dataset. Return data frame
    """
    with open('config.json', 'r') as f:
        config = json.load(f)
    dataset_path = os.path.join(os.getcwd(),
                                config['output_folder_path'] + '/' + dataset)
    try:
        return pd.read_csv(dataset_path)
    except AssertionError:
        raise FileNotFoundError('No dataset in data folder!')


def get_model_folder():
    with open('config.json', 'r') as f:
        config = json.load(f)
    model_folder = os.path.join(
        os.getcwd(), config['output_model_path'] + '/')
    if os.path.isdir(model_folder):
        try:
            assert os.access(model_folder, os.W_OK | os.X_OK)
        except AssertionError:
            print(f'{model_folder} cannot be written to')
    else:
        os.umask(0)
        os.makedirs(model_folder)
    return model_folder


def train_model(train=True):
    """
    Train a logistic regression model and put in model folder
    """
    clf = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='auto',
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False)
    df = get_df()
    y = df['exited']
    X = df.drop(['exited', 'corporation'], axis=1)
    model_folder = get_model_folder()
    model_name = 'trainedmodel.pkl'
    model_path = os.path.join(model_folder, model_name)
    if train:
        model = clf.fit(X, y)
        joblib.dump(model, model_path)
        print("Model trained!")
    else:
        try:
            model = joblib.load(model_path)
        except FileNotFoundError:
            raise FileNotFoundError("No model exists! You need to train one!")

# if __name__ == '__main___':
# train_model()
