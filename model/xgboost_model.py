from xgboost import XGBClassifier


def train_model(X_train, y_train):
    model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)
    model.fit(X_train, y_train)
    return model
