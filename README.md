# Attrition
Background
Imagine that you're the Chief Data Scientist at a big company that has 10,000 corporate clients. Your company is extremely concerned about attrition risk: the risk that some of their clients will exit their contracts and decrease the company's revenue. They have a team of client managers who stay in contact with clients and try to convince them not to exit their contracts. However, the client management team is small, and they're not able to stay in close contact with all 10,000 clients.

The company needs you to create, deploy, and monitor a risk assessment ML model that will estimate the attrition risk of each of the company's 10,000 clients. If the model you create and deploy is accurate, it will enable the client managers to contact the clients with the highest risk and avoid losing clients and revenue.

Creating and deploying the model isn't the end of your work, though. Your industry is dynamic and constantly changing, and a model that was created a year or a month ago might not still be accurate today. Because of this, you need to set up regular monitoring of your model to ensure that it remains accurate and up-to-date. You'll set up processes and scripts to re-train, re-deploy, monitor, and report on your ML model, so that your company can get risk assessments that are as accurate as possible and minimize client attrition.

Project Steps Overview
Data ingestion. Automatically check a database for new data that can be used for model training. Compile all training data to a training dataset and save it to persistent storage. Write metrics related to the completed data ingestion tasks to persistent storage.
Training, scoring, and deploying. Write scripts that train an ML model that predicts attrition risk, and score the model. Write the model and the scoring metrics to persistent storage.
Diagnostics. Determine and save summary statistics related to a dataset. Time the performance of model training and scoring scripts. Check for dependency changes and package updates.
Reporting. Automatically generate plots and documents that report on model metrics. Provide an API endpoint that can return model predictions and metrics.
Process Automation. Create a script and cron job that automatically run all previous steps at regular intervals.

The following are the Python files:

training.py, a Python script meant to train an ML model
scoring.py, a Python script meant to score an ML model
deployment.py, a Python script meant to deploy a trained ML model
ingestion.py, a Python script meant to ingest new data
diagnostics.py, a Python script meant to measure model and data diagnostics
reporting.py, a Python script meant to generate reports about model metrics
app.py, a Python script meant to contain API endpoints
wsgi.py, a Python script to help with API deployment
apicalls.py, a Python script meant to call your API endpoints
fullprocess.py, a script meant to determine whether a model needs to be re-deployed, and to call all other Python scripts when needed


The following are the datasets. Each of them is fabricated datasets that have information about hypothetical corporations.

dataset1.csv and dataset2.csv, found in /practicedata/
dataset3.csv and dataset4.csv, found in /sourcedata/
testdata.csv, found in /testdata/
The following are other files that are included in your starter files:

requirements.txt, a text file and records the current versions of all the modules in the scripts
config.json, a data file that contains names of files used for configuration of the Python scripts

# Suggestions
PDF reports
+ In Step 4 "Reporting", you set up a script that generates a plot of a confusion matrix. Instead of outputting just that raw plot, set up a script that generates a pdf file that contains the plot as well as summary statistics and other diagnostics. This enables more complete, quicker reporting that will really make your project stand out.

In order to accomplish this suggestion, you'll need to add to your reporting.py Python script. You may also need to install modules that enable PDF creation, such as the reportlab module. There are many things you could include in a PDF report about your model: you could include the confusion matrix you generate in reporting.py, you could include all of the outputs of API endpoints you created in app.py, and you could also include the model's F1 score (stored in latestscore.txt) and the files that you ingested to train the model (stored in ingestedfiles.txt).

+ Time Trends
Give your scripts the ability to store diagnostics from previous iterations of your model, and generate reports about time trends. For example, show how the percent of NA elements has gone up or down over many weeks or months, or show whether the timing of ingestion and training has increased or decreased.

You could accomplish this suggestion in several different ways. For example, you could create a directory called /olddiagnostics/, and create a script that copied all of your diagnostics outputs to that folder. You could also add timestamps to the filenames of your output files like ingestedfiles.txt and latestscore.txt.

+ Database setup
Instead of writing results and files to .txt and .csv files, write your datasets and records to SQL databases. This will lead to increased performance and reliability.

In order to accomplish this suggestion, you'll have to set up SQL databases in your workspace. You can accomplish this within Python by installing and using the mysql-connector-python module. You could create a new Python script called dbsetup.py that used this module to set up databases. You could set up a database that stored information about ingested files, another one to store information about model scores, and another one to store information about model diagnostics. Then, you would have to alter ingestion.py, scoring.py, and diagnostics.py so that they wrote to these databases every time they were run.
