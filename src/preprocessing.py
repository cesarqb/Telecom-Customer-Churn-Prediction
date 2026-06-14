
import pandas as pd


# ==========================================
# SEPARAR VARIABLES NUMERICAS Y CATEGORICAS
# ==========================================

def separar_variables(df, numeric_vars, categorical_vars):
    """
    Separa variables numéricas y categóricas.

    Parameters
    ----------
    df : pd.DataFrame
    numeric_vars : list
    categorical_vars : list

    Returns
    -------
    tuple
        (num_cols, cat_cols)
    """

    num_cols = df[numeric_vars].copy()
    cat_cols = df[categorical_vars].copy()

    return num_cols, cat_cols



# ==========================================
# OBTENER VARIABLE TARGET
# ==========================================

def obtener_target(df, target_column='Churn'):
    """
    Obtiene la variable objetivo.

    Parameters
    ----------
    df : pd.DataFrame
    target_column : str

    Returns
    -------
    pd.Series
    """

    return df[target_column].copy()



# ==========================================
# CONVERTIR LA TARGET A SU FORMATO CORRECTO
# ==========================================

def target_encoding_binario(
    target,
    positive_class='Yes'
):
    """
    Convierte una variable objetivo binaria
    a formato 0 y 1.

    Parameters
    ----------
    target : pd.Series

    Returns
    -------
    pd.Series
    """

    return target.apply(
        lambda x: 1 if x == positive_class else 0
    )



# ==========================================
# RECODIFICACION DE VARIABLES CATEGORICAS
# ==========================================

def aplicar_one_hot_encoding(df):
    """
    Aplica One Hot Encoding.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    return pd.get_dummies(df)



# ==========================================
# CONSTRUIR DATASET FINAL
# ==========================================

def construir_dataset_final(
    num_cols,
    cat_cols,
    label
):
    """
    Concatena variables numéricas,
    categóricas transformadas y target.

    Parameters
    ----------
    num_cols : pd.DataFrame
    cat_cols : pd.DataFrame
    label : pd.Series

    Returns
    -------
    pd.DataFrame
    """

    return pd.concat(
        [num_cols, cat_cols, label],
        axis=1
    )