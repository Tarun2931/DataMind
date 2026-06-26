from ml_models.preprocessing import preprocess_data

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

def train_classification_models(df, target):
    data = preprocess_data(df, target )
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(),
        "SVC": SVC(),
        "Gaussian Naive Bayes": GaussianNB()

    }

    results = {}
    best_model = None
    best_model_name = ""
    best_f1 = float("-inf")
    best_predictions = None
    roc_data = None
    for name, model in models.items():
        model.fit(X_train,y_train)
        predictions = model.predict(X_test)
        cv_scores = cross_val_score(model,X_train,y_train,cv=5,scoring="f1_weighted")
        cv_mean = cv_scores.mean()
        accuracy = accuracy_score( y_test, predictions)
        precision = precision_score( y_test,predictions,average="weighted")
        recall = recall_score( y_test,predictions,average="weighted")

        f1 = f1_score(y_test,predictions,average="weighted")

        results[name] = {
        "Accuracy": round(accuracy,2),
        "Precision": round(precision,2),
        "Recall": round(recall,2),
        "F1": round(f1,5),
        "CROSS VALIDATION MEAN": round(cv_mean,5)

        }

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name
            best_predictions = predictions
    
    if hasattr(best_model, "predict_proba"):
        probabilities = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test,probabilities)
        roc_auc = auc(fpr,tpr)
        roc_data = {"fpr": fpr.tolist(),"tpr": tpr.tolist(),"auc": roc_auc}        
    
    if hasattr(best_model, "feature_importances_"):

        feature_importance = dict(zip(feature_names,best_model.feature_importances_))
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1],reverse=True))
    else:
        feature_importance = None
    conf_matrix = confusion_matrix(y_test,best_predictions)
    return {
        "results": results,
        "best_model": best_model,
        "best_model_name": best_model_name,
        "feature_names": feature_names,
        "feature_importance": feature_importance,
        "conf_matrix": conf_matrix,
        "roc_data":roc_data

    }