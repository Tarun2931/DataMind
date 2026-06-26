import pandas as pd
import numpy as np
def dataset_summary(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "total_missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numerical_column_count": len(df.select_dtypes(include=np.number).columns),
        "categorical_column_count": len(df.select_dtypes(exclude=np.number).columns)
}


def column_summary(df):
    summary = pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum(),
        "Unique Values": df.nunique()})
    return summary

import pandas as pd
import numpy as np

def statistical_summary(df):

    numerical_df = df.select_dtypes(include=np.number)

    categorical_df = df.select_dtypes(exclude=np.number)

    if len(numerical_df.columns) > 0:

        numerical_summary = numerical_df.describe()

    else:

        numerical_summary = pd.DataFrame(
            {"Message": ["No numerical columns found"]}
        )

    if len(categorical_df.columns) > 0:

        categorical_summary = categorical_df.describe()

    else:

        categorical_summary = pd.DataFrame(
            {"Message": ["No categorical columns found"]}
        )

    return {

        "numerical_summary": numerical_summary,

        "categorical_summary": categorical_summary

    }

def missing_values(df):
    return df.isnull().sum().to_dict()

def duplicate_rows(df):
    return int(df.duplicated().sum())

def numerical_columns(df):
    return df.select_dtypes(include="number").columns.tolist()

def categorical_columns(df):
    return df.select_dtypes(exclude="number").columns.tolist()

def correlation_matrix(df):
    return df.corr(numeric_only=True)

def skewness_summary(df):
    return df.select_dtypes(include="number").skew().to_dict()

def outlier_summary(df):
    outliers = {}
    numerical_cols = df.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        count = df[(df[col] < lower_bound)|(df[col] > upper_bound)].shape[0]
        outliers[col] = count
    return outliers