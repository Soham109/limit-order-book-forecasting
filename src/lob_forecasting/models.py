from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_majority_model():
    """
    Majority-class baseline.
    """
    return DummyClassifier(strategy="most_frequent")


def make_logistic_model():
    """
    Standardized multinomial logistic regression baseline.
    """
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    )


def make_hgb_model():
    """
    Histogram Gradient Boosting classifier for tabular LOB features.
    """
    return HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_leaf_nodes=31,
        random_state=42,
    )


def get_default_model_builders(include_majority=True):
    """
    Return default model builders used in benchmark experiments.
    """
    builders = {}
    if include_majority:
        builders["Majority baseline"] = make_majority_model
    builders["Logistic regression"] = make_logistic_model
    builders["HistGradientBoosting"] = make_hgb_model
    return builders
