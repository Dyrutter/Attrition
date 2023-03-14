from sklearn.metrics import f1_score
import pandas as pd
import joblib
import os
import json


def load_model(config_path='output_folder_path',
               data='finaldata.csv', model_name='trainedmodel.pkl'):
    """
    Score a trained model
    Inputs:
        config_path = key in config.json to use
        data = test data to score
        model_name = name of model
    """
    # Read in config file
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Get Data Frame
    data_path = os.path.join(
        os.getcwd(), config[config_path] + '/' + data)
    df = pd.read_csv(data_path)

    # Get model
    model_path = os.path.join(os.getcwd(), config['output_model_path'] + '/')
    try:
        model = joblib.load(os.path.join(model_path, model_name))
    except FileNotFoundError:
        return "No model exists!"

    # Get predictions and score
    y_true = df['exited']
    X = df.drop(['exited', 'corporation'], axis=1)
    y_pred = model.predict(X)
    return y_true, y_pred, model


def score_model(y_true, y_pred, model, save=True):
    """
    Score a model
    Inputs: y_true = true values of labels
            y_pred = predicted values of labels
            model = classification model
            save = save score to
    """
    with open('config.json', 'r') as f:
        config = json.load(f)
    score = f1_score(y_true, y_pred)
    if save:
        model_path = os.path.join(
            os.getcwd(), config['output_model_path'] + '/')
        score_file = os.path.join(model_path, 'latestscore.txt')
        with open(score_file, 'w') as fp:
            fp.write(str(score))
    return score


if __name__ == '__main__':
    y_true, y_pred, model = load_model()
    score_model(y_true, y_pred, model)
