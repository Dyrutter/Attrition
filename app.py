from flask import Flask
import json
import os
import diagnostics
import scoring


app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

# Adjust variables to load different data or models
data = 'finaldata.csv'
model = 'trainedmodel.pkl'
config_data = 'output_folder_path'  # Access config.json
config_model = 'prod_deployment_path'  # Access config.json
dataset_csv_path = os.path.join(os.getcwd(), config_data + '/')
model_path = os.path.join(os.getcwd(), config_model + '/')


@app.route("/prediction", methods=['GET', 'POST', 'OPTIONS'])
def predict():
    """
    Call prediction function created in diagnostics.py
    """
    preds = diagnostics.model_predictions(model=model,
                                          config_path=config_data, data=data)
    return json.dumps({'predictions': str(preds)})


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def f1_score():
    """
    Check the f1 score of the deployed model
    """
    y_true, y_pred, model = scoring.load_model()
    score = json.dumps(
        {'f1 score': str(scoring.score_model(y_true, y_pred, model))})
    return score  # automatically uses test data


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summary_stats():
    """
    Return a list of means, medians, and std devs for each column
    Jsonify for ease of reading
    """
    summ = diagnostics.dataframe_summary(data=data, config_path=config_data)
    return json.dumps({'Summary Stats: ': str(summ)})


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostic_data():
    """
    Return the timing of data ingestion and model creation, percentages
    of missing data for each input, and a list of outdated packages
    Must be strings so they can be written to file
    Jsonify for ease of reading
    """
    def timing():
        return str(diagnostics.execution_time())

    def missing_data():
        return str(diagnostics.missing_data())

    def outdated():
        return str(diagnostics.outdated_packages_list(write=False))

    return json.dumps({"Timing: ": timing(),
                       "Missing data%: ": missing_data(),
                       "Outdated modules": outdated()})


def run():
    """
    Start up host
    """
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)


if __name__ == "__main__":
    run()
