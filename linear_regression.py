import pandas
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

# Step 2: Get data
housing = pd.read_csv("C:/Users/Chris/PycharmProjects/ESN/train.csv", index_col=0, encoding="ISO-8859-1")
classes = housing['Unusual']
housing = housing.drop('Unusual', axis=1)
housing = housing.drop('CellName', axis=1)
names = housing.columns


sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
scaler = preprocessing.MinMaxScaler()
d = scaler.fit_transform(housing)
#scaled_df = pd.DataFrame(d, columns=names)




stdScaler = preprocessing.StandardScaler()
a = stdScaler.fit_transform(housing)

qtTrans = preprocessing.QuantileTransformer(output_distribution='uniform')
f = qtTrans.fit_transform(housing)

normalizer = preprocessing.Normalizer()
g = normalizer.fit_transform(housing)

powerTrans = preprocessing.PowerTransformer()
h = powerTrans.fit_transform(housing)

scaled_df = pd.DataFrame(d, columns=names)


y = classes
x = scaled_df




x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42, stratify=y)
#x_train, x_test, y_train, y_test = sss.split(x, y)



# Step 3: Create a model and train it
model = RidgeClassifier(class_weight="balanced")
model.fit(x_train, y_train)

# Step 4: Evaluate the model
predictions = model.predict(x_test)
#predictions = pd.Series(model.predict(x), index=x.index)
report = classification_report(y_test, predictions, output_dict=True)
spreadsheet = pandas.DataFrame(report).transpose()
spreadsheet.to_csv("linear.csv")


print(report)

cv_scores = cross_val_score(model, x_train, y_train, cv=10)
print("CV average score: %.2f" % cv_scores.mean())

r_sq = model.score(x_test, y_test)
print(r_sq)



