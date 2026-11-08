from data import get_split
from id3 import ID3Classifier
from metrics import model_eval

Xtr, Xte, ytr, yte = get_split(0)

model=ID3Classifier(n_bins=5)
model.fit(Xtr, ytr)

pred=model.predict(Xte)
results = model_eval(yte, pred)

for metric, value in results.items():
    print(metric, ":", value)