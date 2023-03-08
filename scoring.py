from sklearn.metrics import f1_score
import pandas as pd
import joblib
import os
import json


def score_model(test_data='testdata.csv', model_name='trainedmodel.pkl'):
    """
    Score a trained model
    """
    # Read in config file
    with open('config.json', 'r') as f:
        config = json.load(f)

    test_data_path = os.path.join(
        os.getcwd(), config['test_data_path'] + '/' + test_data)
    df = pd.read_csv(test_data_path)

    model_path = os.path.join(os.getcwd(), config['output_model_path'] + '/')
    model = joblib.load(os.path.join(model_path, model_name))

    y_true = df['exited']
    X = df.drop(['exited', 'corporation'], axis=1)
    y_pred = model.predict(X)
    score = f1_score(y_true, y_pred)

    score_file = os.path.join(model_path, 'latestscore.txt')
    with open(score_file, 'w') as fp:
        fp.write(str(score))
    return score


if __name__ == '__main__':
    score_model()
