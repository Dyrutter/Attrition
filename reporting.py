import joblib
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
import os
import diagnostics


def graph_matrix(model='trainedmodel.pkl',
                 config_path='test_data_path',
                 data='testdata.csv'):
    """
    Create a confusion matrix using the test data and deployed model
    Inputs: model = model filename
            config_path = path to data in json file
            data = test data file name
    """
    # Load in config.json file
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Get data frame and extract test labels
    data_path = os.path.join(
        os.getcwd(), config[config_path] + '/' + data)
    df = pd.read_csv(data_path)
    y = df['exited']

    # Get predictions
    preds = diagnostics.model_predictions(data=data)

    # Get list of class labels from model
    model_path = os.path.join(
        os.getcwd(), config['prod_deployment_path'] + '/' + model)
    clf = joblib.load(model_path)
    classes = clf.classes_

    # Create confusion matrix
    matrix = confusion_matrix(y, preds, labels=classes)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=classes)
    disp.plot()
    plt.tight_layout()

    # Save matrix to output_folder_path
    output_folder = os.path.join(os.getcwd(), config['output_model_path'])
    plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'))
    plt.close()


if __name__ == '__main__':
    graph_matrix()
