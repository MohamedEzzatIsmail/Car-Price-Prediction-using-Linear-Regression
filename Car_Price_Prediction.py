import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle


file = pd.read_csv('car_price_prediction.csv')
file = file[["Price", "Levy", "Manufacturer", "Model", "Prod. year", "Category",
             "Leather interior", "Fuel type", "Engine volume", "Mileage", "Cylinders",
             "Gear box type", "Drive wheels", "Doors", "Wheel", "Color", "Airbags"]]
file.fillna(0, inplace=True)

le = LabelEncoder()
for column in file.columns:
    if file[column].dtype == object:
        file[column] = le.fit_transform(file[column])

x = np.array(file.drop("Price", axis=1))
y = np.array(file["Price"])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

"""
# Train and save best model
best_acc = 0
for _ in range(50):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.0004)
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print("acc :", str(acc))

    if acc > best_acc:
        best_acc = acc
        with open("model.pickle", "wb") as f:
            pickle.dump(model, f)

print("Best accuracy =", best_acc)

"""
p = open('model.pickle', 'rb')
Linear = pickle.load(p)

predict = Linear.predict(x_test)
for x in range(len(predict)):
    print(predict[x], y_test[x])

