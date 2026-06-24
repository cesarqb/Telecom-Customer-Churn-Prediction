# src/train_model.py

import pickle



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