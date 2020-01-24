import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import easygui

style.use("ggplot")

data = pd.read_csv((easygui.fileopenbox()), sep=',')

predict = "code"

data = data[['code', 'value']]
data.drop(data[data['code'] < 58].index, inplace=True)
data.drop(data[data['code'] > 64].index, inplace=True)
data = shuffle(data)

print(data)

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.5)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print("Accuracy: " + str(acc))

best = 0
for _ in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.5)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
#    print("Accuracy: " + str(acc))
#    print("-----")

    if acc > best:
        best = acc
        with open("test.pickle", "wb") as f:
            pickle.dump(linear, f)

pickle_in = open("test.pickle", "rb")
linear = pickle.load(pickle_in)

print("-------------------------")

print("Overall Accuracy: " + str(acc))

print("-------------------------")
# print('Coefficient: \n', linear.coef_)
# print('Intercept: \n', linear.intercept_)
# print("-------------------------")

predictions = linear.predict(x_test)

# for x in range(len(predictions)):
#    print(predictions[x], x_test[x], y_test[x])


plot = "code"
plt.scatter(data['code'], data['value'])
plt.legend([])
plt.xlabel("Code")
plt.ylabel("Blood Glucose Levels (BGL)")

plt.axhspan(70, 180, color='green', alpha=0.5)

plt.show()
