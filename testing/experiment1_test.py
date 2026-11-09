# run this to test: python -m testing.experiment1_test

from data import get_split
from id3 import ID3Classifier
from metrics import model_eval
from experiment1 import rank_features, select_features

Xtrain, Xtest, ytrain, ytest = get_split(0)

rankings = rank_features(Xtrain, ytrain)

feature_options = [5, 10, 15, 30]

metrics = {"accuracy", "recall"}

for num_features in feature_options:
    X_train, X_test = select_features(Xtrain, Xtest, rankings, num_features)
    model = ID3Classifier(n_bins = 5)
    model.fit(X_train, ytrain)
    predictions = model.predict(X_test)
    results = model_eval(ytest, predictions, metrics)
    print("\nNumber of features: ", num_features)
    for metric, value in results.items():
        print(metric, ": ", value)