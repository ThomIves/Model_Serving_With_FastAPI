import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression

# Create the fake feature data
X1 = np.random.uniform(0, 1, 1000).reshape(-1, 1)
X2 = np.random.uniform(0, 1, 1000).reshape(-1, 1)
X3 = np.random.uniform(0, 1, 1000).reshape(-1, 1)

# Group the fake data features
X = np.hstack((X1, X2, X3))

# Use a known model to create the outputs and add noise
Y = 1.0 * X1 + 2.0 * X2 + 3.0 * X3
Y_peak_noise = np.max(Y) * 0.07
Y_noise = np.random.normal(0, 0.07 * Y_peak_noise, 1000).reshape(-1, 1)
Y += Y_noise

# Create train and test data
X_train, X_test, Y_train, Y_test = sklms.train_test_split(
    X, Y, test_size=0.2, random_state=42, shuffle=True)
print(f"X train shape is: {X_train.shape}")
print(f"X test shape is: {X_test.shape}")
print(f"Y train shape is: {Y_train.shape}")
print(f"Y test shape is: {Y_test.shape}")
print()


# Instantiate the model and train it
mod_LR = LinearRegression(fit_intercept=False, copy_X=True)
mod_LR.fit(X_train, Y_train)

# Check the scoring of the training
print(mod_LR.score(X_train, Y_train))
# Check that the coefficients values are close to the ones used above
print(mod_LR.coef_)

# Save the trained model to file
model_file_name = "Linear_Regression_Model.pkl"
with open(model_file_name, 'wb') as f:
    pickle.dump(mod_LR, f)

# Save the test data to file
data_file_name = "Test_Data.npz"
with open(data_file_name, 'wb') as f:
    np.save(f, np.hstack((X_test, Y_test)))

# Also save the X_test values to a csv file
X_test_df = pd.DataFrame(X_test, columns = ['X1','X2','X3'])
X_test_df.to_csv("X_Test_Data.csv")
