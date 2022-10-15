import numpy as np
import sklearn.metrics as sklm
import math
import pickle


# A handy metric function I picked up from a course
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


# Load the model from the file
model_file_name = "Linear_Regression_Model.pkl"
with open(model_file_name, 'rb') as f:
    mod_LR = pickle.load(f)

# Load the test data
data_file_name = "Test_Data.npz"
with open(data_file_name, 'rb') as f:
    feature_label_data = np.load(f)

Break the test data into features and labels (inputs and outputs)
X_test = feature_label_data[:, :-1]
Y_test = feature_label_data[:, -1]

# Perform predictions and measure the model's performance
Y_pred = mod_LR.predict(X_test)
print_metrics(Y_test, Y_pred, 4)
