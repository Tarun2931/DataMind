from ml_models.preprocessing import preprocess_data
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_regression_models(df, target):
    data = preprocess_data(df,target)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor( n_estimators=100,random_state=42),
        "XGBoost": XGBRegressor(),
        "SVR": SVR()
    }

    results = {}
    best_model = None
    best_model_name = ""
    best_r2 = float("-inf")
    best_actual = None
    best_predictions = None
    for name, model in models.items():
        model.fit(X_train,y_train)
        predictions = model.predict( X_test)
        cv_scores = cross_val_score(model,X_train,y_train,cv=5,scoring="r2")

        cv_mean = cv_scores.mean()
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        mae = mean_absolute_error( y_test, predictions)
        r2 = r2_score( y_test, predictions)
        results[name] = {
        "RMSE": round(rmse,2),
        "MAE": round(mae,2),
        "R2": round(r2,5),
        "CROSS VALIDATION MEAN": round(cv_mean,5)
        }

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name
            best_actual = y_test
            best_predictions = predictions
    # Feature Importance only if supported by best_model thats why used hasattr hehehehhehe 

    if hasattr(best_model, "feature_importances_"):

        feature_importance = dict(zip(feature_names,best_model.feature_importances_))
        feature_importance = dict(sorted(feature_importance.items(),key=lambda x: x[1],reverse=True))

    else:
        feature_importance = None
    
    return {

        "results": results,
        "best_model": best_model,
        "best_model_name": best_model_name,
        "feature_names": feature_names,
        "feature_importance": feature_importance,
        "actual_values": y_test.tolist(),
        "predicted_values": best_model.predict(X_test).tolist()
    }