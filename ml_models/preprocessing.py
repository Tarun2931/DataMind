import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    PowerTransformer
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

def remove_duplicates(df):
    return df.drop_duplicates()

def handle_missing_values(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()
    numerical_cols = X_train.select_dtypes(include=np.number).columns
    categorical_cols = X_train.select_dtypes(exclude=np.number).columns
    for col in numerical_cols:
        if X_train[col].isnull().sum() > 0:
            if abs(X_train[col].skew()) > 1:
                imputer = SimpleImputer(strategy="median")

            else:

                imputer = SimpleImputer(strategy="mean")

            imputer.fit(X_train[[col]])

            X_train[[col]] = imputer.transform(X_train[[col]])

            X_test[[col]] = imputer.transform(X_test[[col]])

    for col in categorical_cols:

        if X_train[col].isnull().sum() > 0:

            imputer = SimpleImputer(strategy="most_frequent")

            imputer.fit(X_train[[col]])

            X_train[[col]] = imputer.transform(X_train[[col]])

            X_test[[col]] = imputer.transform(X_test[[col]])

    return X_train, X_test


def encode_features(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()
    binary_cols = []
    multi_cols = []
    ordinal_categories = [
        ["low", "medium", "high"],
        ["small", "medium", "large"],
        ["poor", "average", "good", "excellent"]
    ]
    for col in X_train.select_dtypes(exclude=np.number).columns:
        values = set(X_train[col].astype(str).str.lower().dropna().unique())
        ordinal_found = False
        for categories in ordinal_categories:
            if values.issubset(set(categories)):
                encoder = OrdinalEncoder(categories=[categories])
                encoder.fit(X_train[[col]].astype(str).apply(lambda x: x.str.lower()))
                X_train[[col]] = encoder.transform(X_train[[col]].astype(str).apply(lambda x: x.str.lower()))
                X_test[[col]] = encoder.transform(X_test[[col]].astype(str).apply(lambda x: x.str.lower()))
                ordinal_found = True
                break
        if ordinal_found:
            continue
        elif X_train[col].nunique() == 2:
            binary_cols.append(col)
        else:
            multi_cols.append(col)
    for col in binary_cols:
        encoder = LabelEncoder()
        encoder.fit(X_train[col])
        X_train[col] = encoder.transform(X_train[col])
        X_test[col] = encoder.transform(X_test[col])
    if len(multi_cols) > 0:
        encoder = OneHotEncoder(drop="first",sparse_output=False,handle_unknown="ignore")

        encoded_train = encoder.fit_transform(X_train[multi_cols])

        encoded_test = encoder.transform( X_test[multi_cols])

        encoded_train_df = pd.DataFrame(encoded_train,columns=encoder.get_feature_names_out(multi_cols),index=X_train.index)

        encoded_test_df = pd.DataFrame(encoded_test,columns=encoder.get_feature_names_out(multi_cols),index=X_test.index)

        X_train = X_train.drop(columns=multi_cols)

        X_test = X_test.drop(columns=multi_cols)

        X_train = pd.concat([X_train, encoded_train_df],axis=1)

        X_test = pd.concat([X_test, encoded_test_df],axis=1)

    return X_train, X_test


def handle_outliers(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()
    numerical_cols = X_train.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        X_train[col] = np.clip(X_train[col],lower_bound,upper_bound)

        X_test[col] = np.clip(X_test[col],lower_bound,upper_bound)

    return X_train, X_test


def handle_skewness(X_train, X_test):

    X_train = X_train.copy()
    X_test = X_test.copy()

    numerical_cols = X_train.select_dtypes(include=np.number).columns

    skewness = X_train[numerical_cols].skew()

    skewed_cols = skewness[abs(skewness) > 1].index

    if len(skewed_cols) > 0:

        transformer = PowerTransformer(method="yeo-johnson")

        transformer.fit(X_train[skewed_cols])

        X_train[skewed_cols] = transformer.transform( X_train[skewed_cols])

        X_test[skewed_cols] = transformer.transform(X_test[skewed_cols])

    return X_train, X_test


def scale_features(X_train,X_test,scaler_type="standard"):

    X_train = X_train.copy()
    X_test = X_test.copy()
    numerical_cols = X_train.select_dtypes(include=np.number).columns
    if scaler_type == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    scaler.fit(X_train[numerical_cols])
    X_train[numerical_cols] = scaler.transform( X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform( X_test[numerical_cols])
    return X_train, X_test


def remove_low_variance_features(X_train,X_test):
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(X_train)

    X_train_new = selector.transform( X_train)
    X_test_new = selector.transform(X_test)
    selected_columns = X_train.columns[selector.get_support()]
    X_train_new = pd.DataFrame(X_train_new,columns=selected_columns,index=X_train.index)
    X_test_new = pd.DataFrame(X_test_new, columns=selected_columns, index=X_test.index)

    return X_train_new, X_test_new


def preprocess_data(df,target):

    df = remove_duplicates(df)

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    X_train, X_test = handle_missing_values(X_train,X_test)
    X_train, X_test = encode_features( X_train, X_test)

    X_train, X_test = handle_outliers(X_train,X_test)

    X_train, X_test = handle_skewness(X_train,X_test)

    # X_train, X_test = scale_features(X_train,X_test)

    X_train, X_test = remove_low_variance_features(X_train, X_test)

    feature_names = X_train.columns.tolist()

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names
    }