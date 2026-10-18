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

    def __init__(
        self,
        min_samples_split=2,
        max_depth=5,
        threshold_batch_size=256
    ):

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

        # Important for RTX 4050.
        # Don't create enormous GPU matrices.
        self.threshold_batch_size = threshold_batch_size

        self.root = None

    # =========================================================
    # FIT
    # =========================================================

    def fit(self, X, Y):

        self.X_gpu = cp.asarray(X, dtype=cp.float32)
        self.Y_gpu = cp.asarray(Y)

        print(
            "Training on GPU:",
            cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        )

        self.root = self.build_tree(
            self.X_gpu,
            self.Y_gpu,
            depth=0
        )

        cp.cuda.Stream.null.synchronize()

        return self

    # =========================================================
    # BUILD TREE
    # =========================================================

    def build_tree(self, X, Y, depth):

        n_samples, n_features = X.shape

        # Stopping conditions
        if n_samples < self.min_samples_split:
            return self.make_leaf(Y)

        if depth >= self.max_depth:
            return self.make_leaf(Y)

        # If everything belongs to one class
        if cp.all(Y == Y[0]):
            return self.make_leaf(Y)

        best = self.find_best_split(X, Y)

        if best is None:
            return self.make_leaf(Y)

        if best["gain"] <= 0:
            return self.make_leaf(Y)

        feature = best["feature"]
        threshold = best["threshold"]

        # GPU split
        mask = X[:, feature] <= threshold

        X_left = X[mask]
        Y_left = Y[mask]

        X_right = X[~mask]
        Y_right = Y[~mask]

        left = self.build_tree(
            X_left,
            Y_left,
            depth + 1
        )

        right = self.build_tree(
            X_right,
            Y_right,
            depth + 1
        )

        return Node(
            feature_idx=feature,
            threshold=threshold,
            left=left,
            right=right,
            info_gain=best["gain"]
        )

    # =========================================================
    # FIND BEST SPLIT
    # =========================================================

    def find_best_split(self, X, Y):

        n_samples, n_features = X.shape

        parent_gini = self.gini(Y)

        best_gain = -1.0
        best_feature = None
        best_threshold = None

        for feature in range(n_features):

            values = X[:, feature]

            # Sort once on GPU
            sorted_values = cp.sort(values)

            # Remove duplicate thresholds
            thresholds = cp.unique(sorted_values)

            if len(thresholds) <= 1:
                continue

            # Avoid testing the largest value
            thresholds = thresholds[:-1]

            # Process thresholds in batches
            for start in range(
                0,
                len(thresholds),
                self.threshold_batch_size
            ):

                end = min(
                    start + self.threshold_batch_size,
                    len(thresholds)
                )

                batch = thresholds[start:end]

                # ------------------------------------------------
                # GPU PARALLEL SPLIT
                #
                # shape:
                #
                # batch_size × n_samples
                # ------------------------------------------------

                mask = values[None, :] <= batch[:, None]

                left_count = cp.sum(
                    mask,
                    axis=1
                )

                right_count = n_samples - left_count

                valid = (
                    (left_count > 0)
                    &
                    (right_count > 0)
                )

                if not cp.any(valid):
                    continue

                # ------------------------------------------------
                # Convert labels into binary class indicators
                # ------------------------------------------------

                classes = cp.unique(Y)

                # For binary classification
                if len(classes) == 2:

                    positive = (
                        Y == classes[1]
                    ).astype(cp.int32)

                    # Number of positive samples on left
                    left_positive = (
                        mask @ positive
                    )

                    total_positive = cp.sum(
                        positive
                    )

                    right_positive = (
                        total_positive
                        - left_positive
                    )

                    # ------------------------------------------------
                    # Gini left
                    # ------------------------------------------------

                    p_left = (
                        left_positive
                        / cp.maximum(left_count, 1)
                    )

                    gini_left = (
                        2 * p_left * (1 - p_left)
                    )

                    # ------------------------------------------------
                    # Gini right
                    # ------------------------------------------------

                    p_right = (
                        right_positive
                        / cp.maximum(right_count, 1)
                    )

                    gini_right = (
                        2 * p_right * (1 - p_right)
                    )

                    # ------------------------------------------------
                    # Weighted Gini
                    # ------------------------------------------------

                    weighted_gini = (
                        (left_count / n_samples)
                        * gini_left
                        +
                        (right_count / n_samples)
                        * gini_right
                    )

                    gains = (
                        parent_gini
                        - weighted_gini
                    )

                else:

                    # Generic multiclass fallback
                    gains = self.multiclass_gains(
                        mask,
                        Y,
                        left_count,
                        right_count,
                        parent_gini
                    )

                # Invalid splits
                gains = cp.where(
                    valid,
                    gains,
                    -cp.inf
                )

                # GPU reduction
                local_index = cp.argmax(gains)

                local_gain = float(
                    gains[local_index].get()
                )

                if local_gain > best_gain:

                    best_gain = local_gain

                    best_feature = feature

                    best_threshold = float(
                        batch[local_index].get()
                    )

        if best_feature is None:
            return None

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "gain": best_gain
        }

    # =========================================================
    # MULTICLASS GAIN
    # =========================================================

    def multiclass_gains(
        self,
        mask,
        Y,
        left_count,
        right_count,
        parent_gini
    ):

        classes = cp.unique(Y)

        n_thresholds = mask.shape[0]

        gini_left = cp.zeros(
            n_thresholds,
            dtype=cp.float32
        )

        gini_right = cp.zeros(
            n_thresholds,
            dtype=cp.float32
        )

        for cls in classes:

            class_mask = (
                Y == cls
            ).astype(cp.float32)

            total = cp.sum(
                class_mask
            )

            left_class = (
                mask @ class_mask
            )

            right_class = (
                total - left_class
            )

            p_left = (
                left_class
                / cp.maximum(left_count, 1)
            )

            p_right = (
                right_class
                / cp.maximum(right_count, 1)
            )

            gini_left += p_left ** 2
            gini_right += p_right ** 2

        gini_left = 1 - gini_left
        gini_right = 1 - gini_right

        weighted = (
            (left_count / mask.shape[1])
            * gini_left
            +
            (right_count / mask.shape[1])
            * gini_right
        )

        return parent_gini - weighted

    # =========================================================
    # GINI
    # =========================================================

    def gini(self, Y):

        _, counts = cp.unique(
            Y,
            return_counts=True
        )

        probabilities = (
            counts / len(Y)
        )

        return float(
            1 - cp.sum(
                probabilities ** 2
            )
        )

    # =========================================================
    # LEAF
    # =========================================================

    def make_leaf(self, Y):

        values, counts = cp.unique(
            Y,
            return_counts=True
        )

        index = cp.argmax(counts)

        value = values[index].item()

        return Node(
            value=value
        )

    # =========================================================
    # PREDICT
    # =========================================================

    def predict(self, X):

        X = np.asarray(X)

        predictions = np.empty(
            len(X),
            dtype=np.int32
        )

        for i, sample in enumerate(X):

            predictions[i] = (
                self.make_prediction(
                    sample,
                    self.root
                )
            )

        return predictions

    # =========================================================
    # SINGLE PREDICTION
    # =========================================================

    def make_prediction(
        self,
        x,
        node
    ):

        if node.value is not None:
            return node.value

        if (
            x[node.feature_idx]
            <= node.threshold
        ):

            return self.make_prediction(
                x,
                node.left
            )

        return self.make_prediction(
            x,
            node.right
        )

    # =========================================================
    # DRAW TREE
    # =========================================================

    def draw_tree(
        self,
        node=None,
        depth=0
    ):

        if node is None:
            node = self.root

        indent = "\t" * depth

        if node.value is not None:

            print(
                indent
                + "Leaf: "
                + str(node.value)
            )

            return

        print(
            indent
            + "Feature "
            + str(node.feature_idx)
            + " <= "
            + str(node.threshold)
            + " "
            + "(Gain: "
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
