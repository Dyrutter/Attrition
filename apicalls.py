import requests
import subprocess
import json
import os

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"


with open('config.json', 'r') as f:
    config = json.load(f)
filename = 'apireturns.txt'

# Call each API endpoint and store the responses
response1 = subprocess.run(
    ['curl', 'http://0.0.0.0:8000/prediction'], capture_output=True).stdout
response2 = subprocess.run(
    ['curl', 'http://0.0.0.0:8000/scoring'], capture_output=True).stdout
response3 = requests.get("http://127.0.0.1:8000/summarystats").content
response4 = subprocess.run(
    ['curl', 'http://0.0.0.0:8000/diagnostics'], capture_output=True).stdout

# Save responses to file
save_file = os.path.join(
    os.getcwd(), config['output_model_path'] + '/' + filename)
with open(save_file, 'w') as f:
    f.write(str(response1, 'utf-8'))
    f.write(str(response2, 'utf-8'))
    f.write(str(response3, 'utf-8'))
    f.write(str(response4, 'utf-8'))
