"""Model definitions, training, and hyperparameter tuning."""

from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


def get_candidate_models(scale_pos_weight: float, random_state: int = 42) -> dict:
    """Nine baseline classifiers benchmarked in this project. Class weighting
    is used where supported so the class imbalance in the target doesn't bias
    models toward always predicting "no default".
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        ),
        "k-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(
            probability=True, class_weight="balanced", random_state=random_state
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=random_state, n_jobs=-1
        ),
        "AdaBoost": AdaBoostClassifier(random_state=random_state),
        "Neural Network (MLP)": MLPClassifier(
            random_state=random_state, max_iter=500, early_stopping=True,
            hidden_layer_sizes=(100, 50),
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, objective="binary:logistic",
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=random_state, n_jobs=-1,
        )
    return models


def tune_random_forest(preprocessor, X_train, y_train, cv, scoring="roc_auc", random_state=42):
    """Small grid search over the parameters that matter most for RF on
    tabular data: number of trees, max depth, and leaf size. Kept narrow so
    it runs in minutes; widen it if you have more compute budget.
    """
    pipe = Pipeline([("preprocess", preprocessor),
                      ("clf", RandomForestClassifier(class_weight="balanced",
                                                      random_state=random_state, n_jobs=-1))])
    param_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [8, 16, None],
        "clf__min_samples_leaf": [1, 3],
    }
    search = GridSearchCV(pipe, param_grid, scoring=scoring, cv=cv, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    return search.best_estimator_.named_steps["clf"], search.best_params_, search.best_score_


def tune_xgboost(preprocessor, X_train, y_train, cv, scale_pos_weight, scoring="roc_auc", random_state=42):
    if XGBClassifier is None:
        return None, None, None
    pipe = Pipeline([("preprocess", preprocessor),
                      ("clf", XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                             scale_pos_weight=scale_pos_weight,
                                             random_state=random_state, n_jobs=-1))])
    param_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [3, 5],
        "clf__learning_rate": [0.05, 0.1],
    }
    search = GridSearchCV(pipe, param_grid, scoring=scoring, cv=cv, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    return search.best_estimator_.named_steps["clf"], search.best_params_, search.best_score_


def build_voting_ensemble(tuned_rf, tuned_xgb, random_state=42):
    estimators = [("lr", LogisticRegression(max_iter=1000, class_weight="balanced",
                                             random_state=random_state))]
    if tuned_rf is not None:
        estimators.append(("rf", tuned_rf))
    if tuned_xgb is not None:
        estimators.append(("xgb", tuned_xgb))
    return VotingClassifier(estimators=estimators, voting="soft")


def build_stacking_ensemble(tuned_rf, tuned_xgb, random_state=42):
    base = [("lr_base", LogisticRegression(max_iter=1000, class_weight="balanced",
                                            random_state=random_state))]
    if tuned_rf is not None:
        base.append(("rf_base", tuned_rf))
    if tuned_xgb is not None:
        base.append(("xgb_base", tuned_xgb))
    base.append(("mlp_base", MLPClassifier(random_state=random_state, max_iter=500,
                                            early_stopping=True, hidden_layer_sizes=(100, 50))))
    return StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced",
                                            random_state=random_state),
        stack_method="predict_proba",
        passthrough=True,
        n_jobs=-1,
    )
