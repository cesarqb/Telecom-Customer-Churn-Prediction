# src/train_model.py

import pickle
from sklearn.model_selection import GridSearchCV


# ==========================================
# TRAIN MODEL
# ==========================================

def train_model(
    model,
    X_train,
    y_train
):
    """
    Entrena un modelo de Machine Learning.

    Parameters
    ----------
    model : estimator
        Modelo de Scikit-Learn o XGBoost.

    X_train : pd.DataFrame
    y_train : pd.Series

    Returns
    -------
    estimator
        Modelo entrenado.
    """

    model.fit(
        X_train,
        y_train
    )

    return model



# ==========================================
# GRID SEARCH
# ==========================================

def train_gridsearch_model(
    model,
    param_grid,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
):
    """
    Realiza la búsqueda de hiperparámetros mediante GridSearchCV.

    Parameters
    ----------
    model : estimator
        Modelo de clasificación.

    param_grid : dict
        Diccionario con los hiperparámetros a evaluar.

    X_train : pd.DataFrame

    y_train : pd.Series

    cv : int, default=5

    scoring : str, default="accuracy"

    n_jobs : int, default=-1

    verbose : int, default=1

    Returns
    -------
    best_model : estimator
        Modelo con los mejores hiperparámetros.

    best_params : dict
        Mejores hiperparámetros encontrados.
    """

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose
    )

    grid_search.fit(
        X_train,
        y_train
    )

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params




# ==========================================
# GENERATE PREDICTIONS
# ==========================================

def generate_predictions(
    model,
    X_train,
    X_test
):
    """
    Genera predicciones y probabilidades.

    Parameters
    ----------
    model : estimator
    X_train : pd.DataFrame
    X_test : pd.DataFrame

    Returns
    -------
    tuple
    """

    y_train_pred = model.predict(
        X_train
    )

    y_test_pred = model.predict(
        X_test
    )

    y_train_prob = model.predict_proba(
        X_train
    )

    y_test_prob = model.predict_proba(
        X_test
    )

    return (
        y_train_pred,
        y_test_pred,
        y_train_prob,
        y_test_prob
    )



# ==========================================
# SAVE MODEL
# ==========================================

def save_model(
    model,
    filepath
):
    """
    Guarda un modelo entrenado.

    Parameters
    ----------
    model : estimator
    filepath : str
    """

    with open(
        filepath,
        "wb"
    ) as file:

        pickle.dump(
            model,
            file
        )

    print(
        f"Modelo guardado en: {filepath}"
    )