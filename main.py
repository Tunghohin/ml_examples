import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import os

if __name__ == "__main__":
    dataset = pd.read_csv(os.path.join(".", "dataset", "1000_Companies.csv"))
    dataset.drop("State", axis=1, inplace=True)
    inputs, outputs = dataset.iloc[:, :-1], dataset.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    y_predict = regressor.predict(x_test)

    print(r2_score(y_test, y_predict))
