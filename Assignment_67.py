# create a neural network model to predict wheather a customer will leave a service
# features : age, monthly charges, tenure, number of complents, customer support calls
# load and create the data set, clean the datatset, apply standard scaler, train FNN model, evaluate accurecy

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


X = [
    [25,500,12,1,2],
    [30,700,24,0,1],
    [45,1200,6,5,8],
    [50,1500,5,6,10],
    [28,600,18,1,1],
    [35,800,30,0,0],
    [48,1400,4,7,9],
    [52,1600,3,8,12],
    [27,550,20,0,1],
    [42,1300,8,4,7]
]

Y = [
    0,0,1,1,0,0,1,1,0,1
]

# step 1 : create the dataset
X = np.array(X)
Y = np.array(Y)

# step 2 : clean the dataset
# In this case, the dataset is already clean, so we can skip this step.

# step 3 : apply standard scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# step 4 : split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
print("Training labels shape:", Y_train.shape)
print("Testing labels shape:", Y_test.shape)

print("Model split completed successfully!")

# step 5 : create and train the FNN model
model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
model.fit(X_train, Y_train)

# 0 = customer will stay, 1 = customer will leave

# step 6 : evaluate the model
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print("Model accuracy:", accuracy)

print("Test the model with new data:")

new_customer = [[46,1450,5,6,9]]

scaled_new_customer = scaler.transform(new_customer)
prediction = model.predict(scaled_new_customer)
print("Prediction for new customer:", prediction)

print("0 = customer will stay, 1 = customer will leave")
print("Model evaluation completed successfully!")






