import os
import ast
import ingestion
import joblib
import training
import scoring
import deployment
import reporting
import diagnostics
import json
import apicalls

# Create model directory if one doesn't already exist
model_dir = os.path.join(os.getcwd(), 'models')
if not os.path.isdir(model_dir):
    os.umask(0)
    os.makedirs(model_dir)

# Get configuration json file
with open('config.json', 'r') as f:
    config = json.load(f)

# Get list of dataset files in data folder
data = ingestion.get_csv_files()

# Read txt file containing list of datasets
ingestedfiles = os.path.join(
    os.getcwd(), config['prod_deployment_path'] + '/' + 'ingestedfiles.txt')
with open(ingestedfiles, 'r') as fp:
    dataset_list = ast.literal_eval(fp.read())

# Check if no new data sets have been added
try:
    assert data != dataset_list
except AssertionError:
    print("There are no new datasets!")

# Add all new data sets to data set list & merge the data frames
for data_set in data:
    if data_set not in dataset_list:
        dataset_list.append(data_set)
all_dfs = ingestion.merge_multiple_dataframes(dataset_list)

# Get score of latest model
latest_score = os.path.join(
    os.getcwd(), config['prod_deployment_path'] + '/' + 'latestscore.txt')
with open(latest_score, 'r') as s:
    former_score = ast.literal_eval(s.read())

# Make predictions on new data using latest model and get score
original_model_path = os.path.join(
    os.getcwd(), config['prod_deployment_path'] + '/' + 'trainedmodel.pkl')
original_clf = joblib.load(original_model_path)

y_true = all_dfs['exited']
X = all_dfs.drop(['exited', 'corporation'], axis=1)
preds = diagnostics.model_predictions()  # (#list(original_clf.predict(X))
new_score = scoring.score_model(y_true, preds, original_clf, save=False)

try:
    assert former_score >= new_score
except AssertionError:
    print("The new score is higher! Don't re-train!")

# If drift occurred, train, store, and graph a new model
training.train_model()
deployment.store_model_into_pickle()
reporting.graph_matrix(config_path='output_folder_path', data='finaldata.csv')

# API calls needs a server to run, so need a function to start server


def apicall():
    """
    Get API call if server is running
    """
    try:
        apicalls.run()
    except Exception:
        print("No server running! Start a server using app.py!")


if __name__ == '__main__':
    apicall()
