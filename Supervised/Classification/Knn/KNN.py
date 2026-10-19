import numpy as np
from collections import Counter
class KNearestNeighbors():
    def __init__(self, k = 5):
        self.k = k
        self.data = None
        self.x = None
        self.y = None

    def fit(self, x, y):
        self.x = x
        self.y = np.asarray([y]).reshape(-1, 1)


    def predict(self, x_test):
        pred = [self.predict_class(new_point) for new_point in x_test]
        return np.asarray(pred)

    def ecludian_distance(self, a, b):
        return np.sqrt( np.sum((a-b) ** 2, axis = 1))

    def predict_class(self, new_point):
        auc = np.array([self.ecludian_distance(new_point, self.x)]).reshape(-1, 1)
        new_array = np.concat((self.y , auc), axis=1).tolist()
        new_array.sort(key=lambda x: x[1])
        vote_list = np.array(new_array[:self.k])
        return Counter(vote_list[:,0]).most_common()[0][0]
        


