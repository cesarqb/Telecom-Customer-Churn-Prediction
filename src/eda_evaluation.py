
import numpy as np
import pandas as pd


# ==========================================
# OBTENER VARIABLES POR TIPO
# ==========================================

def obtener_variables(df):
    """
    Obtiene las variables numéricas y categóricas de un DataFrame.

    Parámetros
    ----------
    df : pd.DataFrame

    Retorna
    -------
    tuple
        (numeric_vars, categorical_vars)
    """

    numeric_vars = df.select_dtypes(
        include=['number']
    ).columns.tolist()

    categorical_vars = df.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()

    return numeric_vars, categorical_vars



# ==========================================
# RESUMEN VARIABLES CATEGÓRICAS
# ==========================================

def resumen_variables_categoricas(df):
    """
    Muestra la distribución porcentual de las variables categóricas.

    Parámetros
    ----------
    df : pd.DataFrame
    """

    for column in df.columns:

        if (
            df[column].dtype == 'object'
            or
            df[column].dtype.name == 'category'
        ):

            print(
                f"Resumen de porcentajes para la variable '{column}':\n"
            )

            print(
                df[column]
                .value_counts(normalize=True)
                * 100
            )

            print("\n" + "-" * 50 + "\n")



# ==========================================
# PORCENTAJE DE VALORES NULOS
# ==========================================

def mostrar_nulos(df):
    """
    Muestra el porcentaje de valores nulos
    por cada variable del DataFrame.

    Parámetros
    ----------
    df : pd.DataFrame
    """

    for column in df.columns:

        missing_percentage = (
            df[column]
            .isnull()
            .mean()
            * 100
        )

        print(
            f"{column}: {missing_percentage:.2f}%"
        )



# ==========================================
# IMPUTACIÓN DE OUTLIERS
# ==========================================

def imputar_valores_extremos(
    df,
    variable,
    metodo='media'
):
    """
    Imputa valores extremos en una variable
    utilizando media o mediana.

    Parámetros
    ----------
    df : DataFrame
    variable : str
    metodo : str

    Retorna
    -------
    DataFrame
    """

    if metodo not in ['media', 'mediana']:
        raise ValueError(
            "El método debe ser 'media' o 'mediana'"
        )

    if metodo == 'media':
        valor_imputacion = df[variable].mean()
    else:
        valor_imputacion = df[variable].median()

    limite_inferior = (
        df[variable].mean()
        - 3 * df[variable].std()
    )

    limite_superior = (
        df[variable].mean()
        + 3 * df[variable].std()
    )

    df[variable] = np.where(
        (
            df[variable] < limite_inferior
        )
        |
        (
            df[variable] > limite_superior
        ),
        valor_imputacion,
        df[variable]
    )

    return df


# ==========================================
# IMPUTACIÓN DE VALORES PERDIDOS
# ==========================================

def imputar_valores(
    df,
    variable,
    metodo='media',
    valor_especifico=None
):
    """
    Imputa valores perdidos en una variable.

    Parámetros
    ----------
    df : pd.DataFrame
    variable : str
    metodo : str
        media, mediana, moda o valor_especifico
    valor_especifico : any

    Retorna
    -------
    pd.DataFrame
    """

    if metodo == 'media':

        imputacion = (
            df[variable].mean()
        )

    elif metodo == 'mediana':

        imputacion = (
            df[variable].median()
        )

    elif metodo == 'moda':

        imputacion = (
            df[variable]
            .mode()[0]
        )

    elif metodo == 'valor_especifico':

        if valor_especifico is None:
            raise ValueError(
                "Debe proporcionar un valor específico para la imputación."
            )

        imputacion = valor_especifico

    else:

        raise ValueError(
            "Método de imputación no reconocido. "
            "Use 'media', 'mediana', 'moda' o 'valor_especifico'."
        )

    df[variable].fillna(
        imputacion,
        inplace=True
    )

    return df