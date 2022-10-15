import requests
import pandas as pd
import numpy as np
import json
import sklearn.metrics as sklm
import math


# The same print metrics function
def print_metrics(y_test, y_pred, n_params):
    ## First compute R^2 and the adjusted R^2
    ## Print the usual metrics and the R^2 values
    MSE = sklm.mean_squared_error(y_test, y_pred)
    RMSE = math.sqrt(sklm.mean_squared_error(y_test, y_pred))
    MAE = sklm.mean_absolute_error(y_test, y_pred)
    MedAE = sklm.median_absolute_error(y_test, y_pred)
    r2 = sklm.r2_score(y_test, y_pred)
    r2_adj = (r2 - (n_params - 1) /
        (y_test.shape[0] - n_params) * (1 - r2))
    
    print('Mean Square Error      = ' + str(MSE))
    print('Root Mean Square Error = ' + str(RMSE))
    print('Mean Absolute Error    = ' + str(MAE))
    print('Median Absolute Error  = ' + str(MedAE))
    print('R^2                    = ' + str(r2))
    print('Adjusted R^2           = ' + str(r2_adj))


# Loading X_test values from the CSV
# We pretend that these are new features
X_test = pd.read_csv("X_Test_Data.csv")
X_test = X_test.values

# Load full test data to get Y_test for metrics
data_file_name = "Test_Data.npz"
with open(data_file_name, 'rb') as f:
    feature_label_data = np.load(f)

Y_test = feature_label_data[:, -1]

# Create an empty array to hold predictions from REST API
Y_pred_from_api = []

# For each row of the X_test values
for curr_features in X_test:
    note = curr_features[0]
    data = curr_features[1:].tolist() # Pydantic NO LIKE numpy arrays
    
    # Form the correct data input structure
    features = {
        "note": str(note),
        "data": data
    }
    
    # Use requests.post with API_URL and features in json format
    # to do a post operation and have the REST API run a prediction
    API_URL = "http://127.0.0.1:8000/run_model"
    response = requests.post(API_URL, json=features)
    
    # Use json.loads to convert the json string to a dictionary
    output = json.loads(response.text)["value"]
    # Add the prediction to our Y_pred_from_api array
    Y_pred_from_api.append(output)

# Convert our Y_pred_from_api array to a numpy array
Y_pred = np.array(Y_pred_from_api)
# Check the metrics
print_metrics(Y_test, Y_pred, 4)
