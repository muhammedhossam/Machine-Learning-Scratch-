from KNN import KNearestNeighbors
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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

    
X_train, X_test, y_train, y_test = get_data(n_samples=2000, n_features=20)


model1 = KNeighborsClassifier(n_neighbors=10)
model2 = KNearestNeighbors(k=50)

model1.fit(
    X_train,
    y_train
)

pred = model1.predict(X_test)

acc = np.mean(pred == y_test) * 100


print(f"Accuracy from sklearn: {acc:.2f}%")

model2.fit(
    X_train,
    y_train
)

pred = model2.predict(X_test)

acc = np.mean(pred == y_test) * 100


print(f"Accuracy from our model: {acc:.2f}%")