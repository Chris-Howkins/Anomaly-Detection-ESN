import pandas
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#from reservoirpy.datasets import mackey_glass
#from reservoirpy.observables import rmse, rsquare
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle

# Step 2: Get data
housing = pd.read_csv("C:/Users/Chris/PycharmProjects/ESN/UCI_Dataset/Set_1.txt")
classes = housing['class']
housing = housing.drop('class', axis=1)
names = housing.columns

scaler = preprocessing.MinMaxScaler()
d = scaler.fit_transform(housing)
#scaled_df = pd.DataFrame(d, columns=names)




stdScaler = preprocessing.StandardScaler()
a = stdScaler.fit_transform(housing)


robustScaler = preprocessing.RobustScaler()
e = robustScaler.fit_transform(housing)
#scaled_df = pd.DataFrame(e, columns=names)

qtTrans = preprocessing.QuantileTransformer(output_distribution='normal')
f = qtTrans.fit_transform(housing)
#scaled_df = pd.DataFrame(e, columns=names)

normalizer = preprocessing.Normalizer()
g = normalizer.fit_transform(housing)
#scaled_df = pd.DataFrame(e, columns=names)


powerTrans = preprocessing.PowerTransformer()
h = powerTrans.fit_transform(housing)



scaled_df = pd.DataFrame(d, columns=names)

#data = scaled_df.drop('class', axis=1)
scaled_df.plot(subplots=True, layout=(13, 2))
plt.show()


y = classes
x = scaled_df


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42, stratify=y)
#x_train, y_train = shuffle(x_train, y_train)
#print(y_train)


# Step 3: Create a model and train it
model = LogisticRegression(class_weight="balanced")
model.fit(x_train, y_train)

# Step 4: Evaluate the model
predictions = model.predict(x_test)
report = classification_report(y_test, predictions, output_dict=True)
spreadsheet = pandas.DataFrame(report).transpose()
spreadsheet.to_csv("logistic.csv")


print(report)

"""
X = mackey_glass(n_timesteps=2000)
from reservoirpy.nodes import Reservoir, Ridge

reservoir = Reservoir(units=100, lr=0.3, sr=1.25)
readout = Ridge(output_dim=1, ridge=1e-5)
esn = reservoir >> readout
esn.fit(X[:500], X[1:501], warmup=100)
predictions = esn.fit(X[:500], X[1:501]).run(X[501:-1])
print("RMSE:", rmse(X[502:], predictions), "R^2 score:", rsquare(X[502:], predictions))




"""

