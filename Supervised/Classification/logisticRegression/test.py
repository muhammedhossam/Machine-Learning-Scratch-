from lg import LogisticRegression

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification



def get_data(n_samples = 200, n_features = 5, test_size = 0.2):

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,
        n_classes=2,
        random_state=157
    )

    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=157
    )
    return  X_train, X_test, y_train, y_test

    
X_train, X_test, y_train, y_test = get_data(n_samples=200000, n_features=20)

model = LogisticRegression()

model.fit(
    X_train,
    y_train
)

pred = model.predict(X_test)

acc = np.mean(pred == y_test) * 100

print(f"Accuracy: {acc:.2f}%")
