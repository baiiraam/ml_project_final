import numpy as np
import pandas as pd
import src.utils.bias_variance_helper as helper


class DummyModel:
    def __init__(
        self,
        probabilities: np.ndarray,
        classes: np.ndarray,
    ) -> None:
        self.probabilities = probabilities
        self.classes_ = classes

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        return self.probabilities[: len(X)]


class DummyAdaBoost:
    def __init__(
        self,
        n_estimators: int,
        learning_rate: float,
        criterion: str,
        random_state: int,
    ) -> None:
        self.classes_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "DummyAdaBoost":
        self.classes_ = np.unique(y)
        return self

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        positive = np.clip(
            0.2 + 0.6 * X[:, 0],
            0.0,
            1.0,
        )

        return np.column_stack([1.0 - positive, positive])

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        probabilities = self.predict_proba(X)[:, 1]

        return np.where(
            probabilities >= 0.5,
            self.classes_[1],
            self.classes_[0],
        )


class DummyRandomForest:
    def __init__(
        self,
        n_estimators: int,
        max_depth,
        max_features,
        min_samples_split: int,
        criterion: str,
        bootstrap: bool,
        oob_score: bool,
        n_jobs: int,
        random_state: int,
    ) -> None:
        self.classes_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "DummyRandomForest":
        self.classes_ = np.unique(y)
        return self

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        positive = np.clip(
            0.3 + 0.4 * X[:, 0],
            0.0,
            1.0,
        )

        return np.column_stack([1.0 - positive, positive])

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        probabilities = self.predict_proba(X)[:, 1]

        return np.where(
            probabilities >= 0.5,
            self.classes_[1],
            self.classes_[0],
        )


def test_bootstrap_sample() -> None:
    X = np.arange(20).reshape(10, 2)
    y = np.array([0, 1] * 5)

    X_bootstrap, y_bootstrap = helper.generate_bootstrap_sample(
        X,
        y,
        random_state=42,
    )

    assert X_bootstrap.shape == X.shape
    assert y_bootstrap.shape == y.shape
    assert set(np.unique(y_bootstrap)) == {0, 1}

    X_second, y_second = helper.generate_bootstrap_sample(
        X,
        y,
        random_state=42,
    )

    assert np.array_equal(
        X_bootstrap,
        X_second,
    )

    assert np.array_equal(
        y_bootstrap,
        y_second,
    )


def test_positive_class_probabilities() -> None:
    probabilities = np.array(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.1, 0.9],
        ]
    )

    model = DummyModel(
        probabilities,
        np.array([0, 1]),
    )

    result = helper.get_positive_class_probabilities(
        model,
        np.zeros((3, 2)),
        positive_class=1,
    )

    expected = np.array([0.2, 0.7, 0.9])

    assert np.allclose(
        result,
        expected,
    )


def test_class_order() -> None:
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.4, 0.6],
        ]
    )

    model = DummyModel(
        probabilities,
        np.array([1, 0]),
    )

    result = helper.get_positive_class_probabilities(
        model,
        np.zeros((2, 1)),
        positive_class=1,
    )

    assert np.allclose(
        result,
        np.array([0.9, 0.4]),
    )


def test_bias_variance_exact_values() -> None:
    predictions = np.array(
        [
            [0.2, 0.8],
            [0.4, 0.6],
        ]
    )

    y_test = np.array([0, 1])

    result = helper.calculate_bias_variance(
        predictions,
        y_test,
    )

    assert np.isclose(
        result["bias_squared"],
        0.09,
    )

    assert np.isclose(
        result["variance"],
        0.01,
    )

    assert np.isclose(
        result["total_error"],
        0.10,
    )

    assert np.isclose(
        result["bias_plus_variance"],
        0.10,
    )

    assert result["difference"] < 1e-12


def test_bias_variance_decomposition() -> None:
    rng = np.random.RandomState(42)

    predictions = rng.uniform(
        0.0,
        1.0,
        size=(50, 20),
    )

    y_test = rng.randint(
        0,
        2,
        size=20,
    )

    result = helper.calculate_bias_variance(
        predictions,
        y_test,
    )

    assert np.isclose(
        result["total_error"],
        result["bias_squared"] + result["variance"],
        atol=1e-12,
    )

    assert result["difference"] < 1e-12


def test_zero_variance() -> None:
    prediction = np.array([0.2, 0.7, 0.9])

    predictions = np.tile(
        prediction,
        (10, 1),
    )

    result = helper.calculate_bias_variance(
        predictions,
        np.array([0, 1, 1]),
    )

    assert np.isclose(
        result["variance"],
        0.0,
        atol=1e-12,
    )

    assert np.isclose(
        result["total_error"],
        result["bias_squared"],
        atol=1e-12,
    )


def test_multiclass_error() -> None:
    X = np.array(
        [
            [0.0],
            [0.2],
            [0.4],
            [0.6],
            [0.8],
            [1.0],
        ]
    )

    y = np.array([0, 1, 2, 0, 1, 2])

    error_raised = False

    try:
        helper.run_bias_variance_experiment(
            X_train=X,
            y_train=y,
            X_test=X,
            y_test=y,
            n_bootstrap=2,
            n_estimators=2,
        )
    except ValueError:
        error_raised = True

    assert error_raised


def test_full_experiment() -> None:
    original_adaboost = helper.AdaBoostClassifier

    original_random_forest = helper.RandomForestClassifier

    helper.AdaBoostClassifier = DummyAdaBoost
    helper.RandomForestClassifier = DummyRandomForest

    try:
        X_train = np.array(
            [
                [0.0],
                [0.1],
                [0.2],
                [0.3],
                [0.7],
                [0.8],
                [0.9],
                [1.0],
            ]
        )

        y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        X_test = np.array(
            [
                [0.1],
                [0.4],
                [0.6],
                [0.9],
            ]
        )

        y_test = np.array([0, 0, 1, 1])

        results = helper.run_bias_variance_experiment(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_bootstrap=5,
            n_estimators=3,
            random_state=42,
            rf_n_jobs=1,
        )

        assert isinstance(
            results,
            pd.DataFrame,
        )

        assert results.shape == (2, 8)

        assert results["model"].tolist() == [
            "AdaBoost",
            "Random Forest",
        ]

        assert np.allclose(
            results["total_error"],
            results["bias_plus_variance"],
            atol=1e-12,
        )

        assert np.all(results["difference"] < 1e-12)

        assert np.all(results["bias_squared"] >= 0)

        assert np.all(results["variance"] >= 0)

        assert np.all(
            results["mean_accuracy"].between(
                0.0,
                1.0,
            )
        )

        assert np.all(
            results["ensemble_accuracy"].between(
                0.0,
                1.0,
            )
        )

    finally:
        helper.AdaBoostClassifier = original_adaboost

        helper.RandomForestClassifier = original_random_forest


def run_all_tests() -> None:
    tests = [
        test_bootstrap_sample,
        test_positive_class_probabilities,
        test_class_order,
        test_bias_variance_exact_values,
        test_bias_variance_decomposition,
        test_zero_variance,
        test_multiclass_error,
        test_full_experiment,
    ]

    for test in tests:
        test()
        print(f"PASSED: {test.__name__}")

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_all_tests()
