# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
**Developed by: SHARON CLARA A**

**RegisterNumber:  212224040310**
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def linear_regression(X1, y, learning_rate=0.1, num_iters=1000):
    X = np.c_[np.ones(len(X1)), X1]   # add bias column
    theta = np.zeros((X.shape[1], 1)) # initialize parameters
    
    for _ in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= learning_rate * (1/len(X1)) * X.T.dot(errors)
    
    return theta


data = pd.read_csv("50_Startups.csv")
print(data.head())
print("\n")


X = data.iloc[:, :-2].values   # take all rows, exclude last 2 cols
y = data.iloc[:, -1].values.reshape(-1, 1)  # target


scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

print("Original Features:\n", X[:5])
print("\nScaled Features:\n", X_scaled[:5])
print("\n")


theta = linear_regression(X_scaled, y_scaled)
print("Learned Parameters (theta):\n", theta)


new_data = np.array([[165349.2, 136897.8, 471784.1]])   
new_scaled = scaler_X.transform(new_data)               
prediction_scaled = np.dot(np.append(1, new_scaled).reshape(1, -1), theta)


prediction = scaler_y.inverse_transform(prediction_scaled)
print("\nScaled Prediction:", prediction_scaled)
print(f"Predicted Profit: {prediction[0][0]}")


```

## Output:

<img width="680" height="667" alt="484051548-f075d3e6-4141-4afd-a2a3-0e8cf95b4572" src="https://github.com/user-attachments/assets/18c5b0f6-636b-473d-9c3a-d5918f34c62e" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
