from id3_gpu import DecisionTree

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


X, y = make_classification(
    n_samples=200,
    n_features=10,
    n_informative=8,
    n_classes=2,
    random_state=157
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=157
)

model = DecisionTree(
    max_depth=3,
    threshold_batch_size=256
)

model.fit(
    X_train,
    y_train
)

pred = model.predict(X_test)

acc = np.mean(
    pred == y_test
) * 100

print(
    f"Accuracy: {acc:.2f}%"
)

model.draw_tree()
