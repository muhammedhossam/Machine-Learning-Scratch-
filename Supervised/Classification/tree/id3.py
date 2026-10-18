import numpy as np
import cupy as cp


class Node:
    def __init__(
        self,
        feature_idx=None,
        threshold=None,
        left=None,
        right=None,
        info_gain=None,
        value=None
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value


class DecisionTree:

    def __init__(self, min_samples_split=2, max_depth=2):

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    # --------------------------------------------------
    # BUILD TREE
    # --------------------------------------------------

    def build_tree(self, dataset, curr_depth=0):

        X = dataset[:, :-1]
        Y = dataset[:, -1]

        n_samples, n_features = X.shape

        if (
            n_samples >= self.min_samples_split
            and curr_depth < self.max_depth
        ):

            best_split = self.best_split(
                dataset,
                n_features
            )

            if best_split["info_gain"] > 0:

                left_subtree = self.build_tree(
                    best_split["dataset_left"],
                    curr_depth + 1
                )

                right_subtree = self.build_tree(
                    best_split["dataset_right"],
                    curr_depth + 1
                )

                return Node(
                    feature_idx=best_split["feature_idx"],
                    threshold=best_split["threshold"],
                    left=left_subtree,
                    right=right_subtree,
                    info_gain=best_split["info_gain"]
                )

        # Leaf node
        leaf_value = self.majority_class(Y)

        return Node(value=leaf_value)

    # --------------------------------------------------
    # BEST SPLIT
    # --------------------------------------------------

    def best_split(self, dataset, n_features):

        best_split = {
            "feature_idx": None,
            "threshold": None,
            "dataset_left": None,
            "dataset_right": None,
            "info_gain": -1
        }

        for feature_idx in range(n_features):

            feature_values = dataset[:, feature_idx]

            # GPU
            thresholds = cp.unique(feature_values)

            for threshold in thresholds:

                # GPU boolean masks
                left_mask = feature_values <= threshold
                right_mask = feature_values > threshold

                dataset_left = dataset[left_mask]
                dataset_right = dataset[right_mask]

                if len(dataset_left) == 0 or len(dataset_right) == 0:
                    continue

                info_gain = self.information_gain(
                    dataset,
                    dataset_left,
                    dataset_right
                )

                if info_gain > best_split["info_gain"]:

                    best_split["feature_idx"] = feature_idx

                    # Convert only this small scalar to CPU
                    best_split["threshold"] = float(
                        threshold.get()
                    )

                    best_split["dataset_left"] = dataset_left

                    best_split["dataset_right"] = dataset_right

                    best_split["info_gain"] = info_gain

        return best_split

    # --------------------------------------------------
    # INFORMATION GAIN
    # --------------------------------------------------

    def information_gain(self, parent, left, right):

        weight_left = len(left) / len(parent)
        weight_right = len(right) / len(parent)

        gain = (
            self.impurity(parent)
            -
            (
                weight_left * self.impurity(left)
                +
                weight_right * self.impurity(right)
            )
        )

        return float(gain)

    # --------------------------------------------------
    # GINI IMPURITY
    # --------------------------------------------------

    def impurity(self, dataset):

        Y = dataset[:, -1]

        if len(Y) == 0:
            return 0

        # GPU
        _, counts = cp.unique(
            Y,
            return_counts=True
        )

        probabilities = counts / len(Y)

        impurity = 1 - cp.sum(
            probabilities ** 2
        )

        return float(impurity)

    # --------------------------------------------------
    # MAJORITY CLASS
    # --------------------------------------------------

    def majority_class(self, Y):

        values, counts = cp.unique(
            Y,
            return_counts=True
        )

        index = cp.argmax(counts)

        return values[index].item()

    # --------------------------------------------------
    # FIT
    # --------------------------------------------------

    def fit(self, X, Y):

        # Move data to GPU
        X_gpu = cp.asarray(X)
        Y_gpu = cp.asarray(Y)

        # Concatenate on GPU
        dataset = cp.concatenate(
            (
                X_gpu,
                Y_gpu.reshape(-1, 1)
            ),
            axis=1
        )

        print(
            f"Training on GPU: "
            f"{cp.cuda.Device().id}"
        )

        self.root = self.build_tree(dataset)

        # Wait for GPU calculations
        cp.cuda.Stream.null.synchronize()

        return self

    # --------------------------------------------------
    # PREDICT
    # --------------------------------------------------

    def predict(self, X):

        X_gpu = cp.asarray(X)

        predictions = [
            self.make_prediction(x, self.root)
            for x in X_gpu
        ]

        return np.array(predictions)

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    def make_prediction(self, x, node):

        if node.value is not None:
            return node.value

        feature_val = x[node.feature_idx]

        if float(feature_val) <= node.threshold:

            return self.make_prediction(
                x,
                node.left
            )

        else:

            return self.make_prediction(
                x,
                node.right
            )

    # --------------------------------------------------
    # DRAW TREE
    # --------------------------------------------------

    def draw_tree(self, node=None, depth=0):

        if node is None:
            node = self.root

        if node.value is not None:

            print(
                "\t" * depth
                + "Leaf: "
                + str(node.value)
            )

        else:

            print(
                "\t" * depth
                + "Feature "
                + str(node.feature_idx)
                + " <= "
                + str(node.threshold)
                + " "
                + "(Info Gain: "
                + str(node.info_gain)
                + ")"
            )

            self.draw_tree(
                node.left,
                depth + 1
            )

            self.draw_tree(
                node.right,
                depth + 1
            )
