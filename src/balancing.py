
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from imblearn.over_sampling import SMOTE



# ==========================================
# TRAIN TEST SPLIT
# ==========================================

def train_test_split_data(
    df,
    target='Churn',
    test_size=0.3,
    random_state=10
):
    """
    Realiza la división Train/Test.

    Parameters
    ----------
    df : pd.DataFrame
    target : str
    test_size : float
    random_state : int

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test



# ==========================================
# OVERSAMPLING
# ==========================================

def oversampling(
    X_train,
    y_train,
    n_samples=None,
    random_state=42
):
    """
    Aplica Oversampling a la clase minoritaria.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series
    n_samples : int, optional
        Cantidad de muestras deseadas para la clase minoritaria.
        Si es None, iguala el tamaño de la clase mayoritaria.
    random_state : int

    Returns
    -------
    pd.DataFrame
    """

    data_train = pd.concat(
        [X_train, y_train],
        axis=1
    )

    target_name = y_train.name

    majority = data_train[
        data_train[target_name] == 0
    ]

    minority = data_train[
        data_train[target_name] == 1
    ]

    if n_samples is None:
        n_samples = len(majority)

    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=n_samples,
        random_state=random_state
    )

    upsample_data_complete = pd.concat(
        [majority, minority_upsampled]
    )

    return upsample_data_complete



# ==========================================
# UNDERSAMPLING
# ==========================================

def undersampling(
    X_train,
    y_train,
    n_samples=None,
    random_state=42
):
    """
    Aplica Undersampling a la clase mayoritaria.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series
    n_samples : int, optional
        Cantidad de muestras deseadas para la clase mayoritaria.
        Si es None, iguala el tamaño de la clase minoritaria.
    random_state : int

    Returns
    -------
    pd.DataFrame
    """

    data_train = pd.concat(
        [X_train, y_train],
        axis=1
    )

    target_name = y_train.name

    majority = data_train[
        data_train[target_name] == 0
    ]

    minority = data_train[
        data_train[target_name] == 1
    ]

    if n_samples is None:
        n_samples = len(minority)

    majority_undersampled = resample(
        majority,
        replace=False,
        n_samples=n_samples,
        random_state=random_state
    )

    undersample_data_complete = pd.concat(
        [majority_undersampled, minority]
    )

    return undersample_data_complete



# ==========================================
# SMOTE
# ==========================================

def aplicar_smote(
    X_train,
    y_train,
    sampling_strategy=0.6,
    random_state=42
):
    """
    Aplica balanceo utilizando SMOTE.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series
    sampling_strategy : float
    random_state : int

    Returns
    -------
    X_train_smote, y_train_smote
    """

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )

    return X_train_smote, y_train_smote