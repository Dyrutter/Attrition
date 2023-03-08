import pandas as pd
import os
import json
import logging

# Create log file if one doesn't already exist and add to log with each run
# Mk separate log files, basic_config only configures unconfigured root files
logging.basicConfig(
    filename='./logs',  # Path to log file
    level=logging.INFO,  # Log info, warnings, errors, and critical errors
    filemode='a',
    format='%(asctime)s-%(name)s - %(levelname)s - %(message)s',
    datefmt='%d %b %Y %H:%M:%S %Z',
    force=True)

# Access the created logger
logger = logging.getLogger()

# Load config.json and get input and output paths
# Config file must be stored in same dir as ingestion.py
with open('config.json', 'r') as f:
    config = json.load(f)


def get_csv_files():
    """
    Get list of csv files in data directory and return their full path.
    """
    # Get full path to csv folder and the csv files inside
    input_folder_path = os.path.join(
        os.getcwd(), config['input_folder_path'] + '/')
    input_data_files = os.listdir(input_folder_path)
    csv_files = [f for f in input_data_files if str('.csv') in f]
    # Return list of csv filepaths
    return [os.path.join(
            input_folder_path, csv_file) for csv_file in csv_files]


def output_folder():
    """
    Get path to output folder. Create one if it doesn't already exist.
    """
    # Make ingested data directory if one doesn't exist or is not writeable
    output_folder_path = os.path.join(
        os.getcwd(), config['output_folder_path'] + '/')
    if os.path.isdir(output_folder_path):
        try:
            # Confirm folder can be written to
            assert os.access(output_folder_path, os.W_OK | os.X_OK)
        except AssertionError:
            print(f'{output_folder_path} cannot be written to')

    # Create writeable folder if one doesn't exist
    else:
        os.umask(0)  # Ensure folder can be written to
        os.makedirs(output_folder_path)
    return output_folder_path


def merge_multiple_dataframe():
    """
    Merge multiple data frames and write the output to a new csv file
    Shapes and Column names must be the same in all instances
    Write the names of the csv file files used to ingestedfiles.txt
    """
    # Merge data frames and drop duplicates
    csv_files = get_csv_files()
    full_df = pd.DataFrame()
    for csv in csv_files:
        df = pd.read_csv(csv)
        full_df = pd.concat([df, full_df])
    df = full_df.drop_duplicates()

    # Save csv file to output folder
    output_csv_folder = output_folder()
    output_csv = 'finaldata.csv'
    df.to_csv(os.path.join(output_csv_folder, output_csv), index=False)
    logger.info(f'{output_csv} Data Frame file has been created')

    # Save list of used data frames to output folder
    used_csvs = os.path.join(output_csv_folder, 'ingestedfiles.txt')
    with open(used_csvs, 'w') as fp:
        fp.write(str(csv_files))
    logger.info(f'Data Frames from {csv_files} were used')
    return df


if __name__ == '__main__':
    df = merge_multiple_dataframe()
    print(df.shape)
