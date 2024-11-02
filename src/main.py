# design an MNIST machine learning model

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

# load data set

digits = load_digits()
X = digits.data
y = digits.target

model = RandomForestClassifier()

# train model

model.fit(X,y)

results = model.predict(X)
accuracy = accuracy_score(y,results)
print("Accuracy= ",accuracy)
precision = precision_score(y,results, average='macro')
print("precision= ",precision)