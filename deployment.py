import shutil
import os
import json


def store_model_into_pickle():
    """
    Copy model, model score, and ingested data to production directory
    """
    # Load config.json
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Paths to files
    model = os.path.join(
        os.getcwd(),
        config['output_model_path'] +
        '/' +
        'trainedmodel.pkl')
    score = os.path.join(
        os.getcwd(),
        config['output_model_path'] +
        '/' +
        'latestscore.txt')
    data_files = os.path.join(
        os.getcwd(),
        config['output_folder_path'] +
        '/' +
        'ingestedfiles.txt')
    prod_deployment_path = os.path.join(
        os.getcwd(), config['prod_deployment_path'] + '/')

    # Create production directory if one doesn't already exist
    if not os.path.exists(prod_deployment_path):
        os.umask(0)
        os.makedirs(prod_deployment_path)

    # Copy files to production path
    shutil.copy(model, prod_deployment_path)
    shutil.copy(score, prod_deployment_path)
    shutil.copy(data_files, prod_deployment_path)


if __name__ == '__main__':
    store_model_into_pickle()
