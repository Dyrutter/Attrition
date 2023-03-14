import pandas as pd
import timeit
import subprocess
import joblib
import os
import json


def model_predictions(
        model='trainedmodel.pkl',
        config_path='output_folder_path',
        data='finaldata.csv'):
    """
    Get list of model predictions on test data
    Inputs: Model = prediction model
            config_path = data folder to use, specified in config.json
            data = test dataset
    """
    # Load model and get classifier
    with open('config.json', 'r') as f:
        config = json.load(f)
    model_path = os.path.join(
        os.getcwd(), config['prod_deployment_path'] + '/' + model)
    clf = joblib.load(model_path)

    # Get test data and put in data frame
    data_path = os.path.join(
        os.getcwd(), config[config_path] + '/' + data)
    df = pd.read_csv(data_path)

    # Separate features from labels and predict on labels
    X = df.drop(['exited', 'corporation'], axis=1)
    return list(clf.predict(X))


def dataframe_summary(data='finaldata.csv', config_path='output_folder_path'):
    """
    Return a nested list of summary statistics where list one = means,
    list two = medians list three = standard deviations.
    Input: test_data = data frame to analyze
    """
    with open('config.json', 'r') as f:
        config = json.load(f)
    data_path = os.path.join(
        os.getcwd(), config[config_path] + '/' + data)
    df = pd.read_csv(data_path)

    # Get features and find mean, median, and std dev
    X = df.drop(['exited', 'corporation'], axis=1)
    metric_list = []
    means = list(X.mean())
    means = f'Means: {means}'
    medians = list(X.median())
    medians = f'Medians: {medians}'
    std = list(X.std())
    std = f'Standard Deviations: {std}'
    metric_list.append(means)
    metric_list.append(medians)
    metric_list.append(std)
    return metric_list


def missing_data(data='finaldata.csv'):
    """
    Get percentage of missing data in each column
    Input: Data to scrutinize
    """
    with open('config.json', 'r') as f:
        config = json.load(f)
    data_path = os.path.join(
        os.getcwd(), config['output_folder_path'] + '/' + data)

    df = pd.read_csv(data_path)
    nas = (list(df.isna().sum()))
    na_percents = [nas[i] / len(df.index) for i in range(len(nas))]
    return na_percents


def execution_time():
    """
    Calculate run time for data ingestion and model training.
    Return a list of 2 timing values in seconds
    """
    # Get paths to python files
    ingestion_path = os.path.join(os.getcwd(), 'ingestion.py')
    training_path = os.path.join(os.getcwd(), 'training.py')

    # Function for timing data ingestion
    def ingestion():
        starttime = timeit.default_timer()
        subprocess.check_output(['python', str(ingestion_path)])
        timing = timeit.default_timer() - starttime
        return timing

    # Function for timing model training
    def training():
        starttime = timeit.default_timer()
        subprocess.check_output(['python', str(training_path)])
        timing = timeit.default_timer() - starttime
        return timing
    return [ingestion(), training()]


def outdated_packages_list(write=True):
    """
    Check current and latest versions of all modules and dependencies
    Write to requirements text file if write=True
    """
    requirements = os.path.join(os.getcwd(), 'requirements.txt')
    if write:
        with open(requirements, 'wb+') as f:
            f.write(subprocess.check_output(
                ['pip', 'list', '--outdated', '>', 'requirements.txt']))
    else:
        return subprocess.check_output(
            ['pip', 'list', '--outdated', '>', 'requirements.txt'])


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()
