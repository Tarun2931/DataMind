from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.templating import Jinja2Templates
import pandas as pd
from ml_models.problem_detection import detect_problem
from ml_models.regression_models import train_regression_models
from ml_models.classification_models import train_classification_models
from ml_models.prediction import predict_value
from ml_models.exploratory_data_analysis import (
    numerical_columns as get_numerical_columns,
    categorical_columns as get_categorical_columns,
    dataset_summary,
    column_summary,
    statistical_summary,
    missing_values,
    duplicate_rows,
    skewness_summary,
    outlier_summary
)
from fastapi.staticfiles import StaticFiles
from graphs.histogram import create_histogram
from graphs.boxplot import create_boxplot
from graphs.scatterplot import create_scatter_plot
from graphs.heatmap import create_heatmap
from graphs.countplot import create_countplot
from ml_models.preprocessing import preprocess_data
from graphs.roc_curve import create_roc_curve
from graphs.actual_vs_predicted import create_actual_vs_predicted
from graphs.residual import create_residual_plot
from ml_models.llm import ask_llm

processed_df = None
feature_importance = None
conf_matrix = None
actual_values = None
predicted_values = None

app = FastAPI()
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

templates = Jinja2Templates(directory="templates")


current_df = None
roc_data = None
numerical_cols = []
categorical_cols = []
trained_model = None
feature_names = []
results = {}
best_model_name = ""
eda_info = {}
chat_context = ""
chat_history = []
processed_train = None
prediction_feature_types = {}
# Home page hai ye




@app.get("/")
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "eda_info": None
        }
    )


# Analyze Dataset wala part




@app.post("/analyze")
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    target: str = Form(...)
):

    global current_df
    global numerical_cols
    global categorical_cols
    global trained_model
    global feature_names
    global results
    global best_model_name
    global eda_info
    global processed_df
    global roc_data
    global feature_importance
    global conf_matrix
    global actual_values
    global predicted_values
    global prediction_feature_types
    global problem_type
    global processed_train
    global chat_context
    

    df = pd.read_csv(file.file)

    df.columns = df.columns.str.strip()

    target = target.strip()

    current_df = df

    numerical_cols = get_numerical_columns(df)

    categorical_cols = get_categorical_columns(df)
    
    eda_info = {

    "dataset_summary": dataset_summary(df),

    "column_summary":
        column_summary(df).to_html(
            classes="table table-dark table-striped table-hover",
            border=1
        ),

    "statistical_summary_num":
        statistical_summary(df)["numerical_summary"].to_html(
            classes="table table-dark table-striped table-hover",
            border=1
        ),

    "statistical_summary_cat":
        statistical_summary(df)["categorical_summary"].to_html(
            classes="table table-dark table-striped table-hover",
            border=1
        ),

    "missing_values": missing_values(df),

    "duplicate_rows": duplicate_rows(df),

    "skewness": skewness_summary(df),

    "outliers": outlier_summary(df)

}

    problem_type = detect_problem(df, target)

    if problem_type == "regression":

        output = train_regression_models(df,target)
        actual_values = output["actual_values"]
        predicted_values = output["predicted_values"]
        print(actual_values)
        print(predicted_values)

    else:

        output = train_classification_models(df,target)
        actual_values=None 
        predicted_values=None
        roc_data = output["roc_data"]

    results = output["results"]

    trained_model = output["best_model"]

    best_model_name = output["best_model_name"]

    feature_names = output["feature_names"]
    feature_importance = output["feature_importance"]
    conf_matrix = output.get("conf_matrix")
    prediction_feature_types = {}

    for feature in feature_names:

        if feature in numerical_cols:

            prediction_feature_types[feature] = "numerical"

        else:

            prediction_feature_types[feature] = "categorical"

    chat_context = f"""
Problem Type:
{problem_type}

Best Model:
{best_model_name}

Model Results:
{results}

Feature Importance:
{feature_importance}

Dataset Summary:
Rows: {df.shape[0]}
Columns: {df.shape[1]}

Missing Values:
{eda_info['missing_values']}

Duplicate Rows:
{eda_info['duplicate_rows']}

Skewness:
{eda_info['skewness']}

Outliers:
{eda_info['outliers']}
"""
      
    

    preprocesse_data = preprocess_data(df,target)
    processed_train = (
    preprocesse_data["X_train"]
    .head(10)
    .to_html(
        classes="table",
        border=1
    )
)

    processed_df = preprocesse_data["X_train"]

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "results": results,
            "best_model_name": best_model_name,
            "feature_names": feature_names,
            "show_insights": True,
            "eda_info": eda_info,
            "processed_train": processed_train,
            "feature_importance": feature_importance,
            "conf_matrix": conf_matrix,
            "problem_type": problem_type,
        }
    )
# Prediction wala part




@app.post("/predict")
async def predict(request: Request):
    global trained_model
    global feature_names
    global results
    global best_model_name
    global prediction_feature_types
    form = await request.form()
    values = []

    for feature in feature_names:

        value = form[feature]

        if prediction_feature_types[feature] == "numerical":

            values.append(float(value))

        else:

            values.append(value)

    prediction = predict_value(trained_model,values)

    return templates.TemplateResponse(
        request=request,
        name="prediction.html",
        context={
    "results": results,
    "best_model_name": best_model_name,
    "feature_names": feature_names,
    "prediction": prediction,
    "show_insights": True,
    "eda_info": eda_info
}
    )



@app.get("/insights")
def insights(request: Request):

    global numerical_cols
    global categorical_cols
    global processed_df

    processed_numerical_cols = []
    processed_categorical_cols = []

    if processed_df is not None:

        processed_numerical_cols = (
            processed_df
            .select_dtypes(include="number")
            .columns
            .tolist()
        )

        processed_categorical_cols = (
            processed_df
            .select_dtypes(exclude="number")
            .columns
            .tolist()
        )

    return templates.TemplateResponse(
        request=request,
        name="insights.html",
        context={

            "numerical_columns": numerical_cols,

            "categorical_columns": categorical_cols,

            "processed_numerical_columns": processed_numerical_cols,

            "processed_categorical_columns": processed_categorical_cols

        }
    )





@app.get("/roc_curve")
def roc_curve_page(request: Request):

    global roc_data

    if roc_data is None:

        return templates.TemplateResponse(
            request=request,
            name="roc_curve.html",
            context={
                "graph_path": None
            }
        )

    fig = create_roc_curve(
        roc_data["fpr"],
        roc_data["tpr"],
        roc_data["auc"]
    )

    fig.savefig(
        "static/graphs/roc_curve.png"
    )

    return templates.TemplateResponse(
        request=request,
        name="roc_curve.html",
        context={
            "graph_path": "/static/graphs/roc_curve.png",
            "auc": round(
                roc_data["auc"],
                4
            )
        }
    )







@app.get("/actual_vs_predicted")
def actual_vs_predicted(request: Request):

    global actual_values
    global predicted_values
    global numerical_cols
    global categorical_cols
    global processed_df

    fig = create_actual_vs_predicted(
        actual_values,
        predicted_values
    )

    fig.savefig(
        "static/graphs/graph.png",
        dpi=300,
        bbox_inches="tight"
    )

    processed_numerical_cols = []
    processed_categorical_cols = []

    if processed_df is not None:

        processed_numerical_cols = (
            processed_df
            .select_dtypes(include="number")
            .columns
            .tolist()
        )

        processed_categorical_cols = (
            processed_df
            .select_dtypes(exclude="number")
            .columns
            .tolist()
        )

    return templates.TemplateResponse(
        request=request,
        name="actual_vs_predicted.html",
        context={
            "graph_path": "/static/graphs/graph.png",

            "numerical_columns": numerical_cols,

            "categorical_columns": categorical_cols,

            "processed_numerical_columns": processed_numerical_cols,

            "processed_categorical_columns": processed_categorical_cols
        }
    )





@app.get("/residual_plot")
def residual_plot(request: Request):

    global actual_values
    global predicted_values

    fig = create_residual_plot(
        actual_values,
        predicted_values
    )

    fig.savefig(
        "static/graphs/residual_plot.png",
        dpi=300,
        bbox_inches="tight"
    )

    return templates.TemplateResponse(
        request=request,
        name="residual.html",
        context={
            "graph_path": "/static/graphs/residual_plot.png"
        }
    )






@app.post("/generate_graph")
async def generate_graph(
    request: Request,
    graph_type: str = Form(...),
    dataset_type: str = Form(...),
    column: str = Form(None),
    category_column: str = Form(None),
    x_column: str = Form(None),
    y_column: str = Form(None)
):

    global current_df
    global processed_df
    global numerical_cols
    global categorical_cols

    if dataset_type == "Raw":

        df_used = current_df

    else:

        df_used = processed_df

    if graph_type == "Histogram":

        fig = create_histogram(
            df_used,
            column
        )

    elif graph_type == "Boxplot":

        fig = create_boxplot(
            df_used,
            column
        )

    elif graph_type == "Scatter Plot":

        fig = create_scatter_plot(
            df_used,
            x_column,
            y_column
        )

    elif graph_type == "Heatmap":

        fig = create_heatmap(
            df_used
        )

    elif graph_type == "Countplot":

        fig = create_countplot(
            df_used,
            category_column
        )

    fig.savefig(
        "static/graphs/graph.png"
    )

    processed_numerical_cols = []
    processed_categorical_cols = []

    if processed_df is not None:

        processed_numerical_cols = (
            processed_df
            .select_dtypes(include="number")
            .columns
            .tolist()
        )

        processed_categorical_cols = (
            processed_df
            .select_dtypes(exclude="number")
            .columns
            .tolist()
        )

    return templates.TemplateResponse(
        request=request,
        name="graph.html",
        context={

            "numerical_columns": numerical_cols,

            "categorical_columns": categorical_cols,

            "processed_numerical_columns": processed_numerical_cols,

            "processed_categorical_columns": processed_categorical_cols,

            "graph_path": "/static/graphs/graph.png",
            "problem_type": problem_type,

        }
    )




@app.get("/chatbot")
def chatbot(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="chatbot.html",
        context={}
    )



@app.post("/ask_ai")
def ask_ai(question: str = Form(...)):

    global chat_context

    answer = ask_llm(
        chat_context,
        question
    )

    return {
        "answer": answer
    }

@app.get("/eda")
def eda(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="eda.html",
        context={
            "eda_info": eda_info,
            "processed_train": processed_train
        }
    )




@app.get("/graphs")
def graphs(request: Request):

    global numerical_cols
    global categorical_cols
    global processed_df
    global problem_type

    processed_numerical_cols = []
    processed_categorical_cols = []

    if processed_df is not None:

        processed_numerical_cols = (
            processed_df
            .select_dtypes(include="number")
            .columns
            .tolist()
        )

        processed_categorical_cols = (
            processed_df
            .select_dtypes(exclude="number")
            .columns
            .tolist()
        )

    return templates.TemplateResponse(
        request=request,
        name="graph.html",
        context={
            "problem_type": problem_type,
            "numerical_columns": numerical_cols,
            "categorical_columns": categorical_cols,
            "processed_numerical_columns": processed_numerical_cols,
            "processed_categorical_columns": processed_categorical_cols,
            "graph_path": None
        }
    )

@app.get("/prediction")
def prediction(request: Request):

    global feature_names
    global prediction_feature_types

    return templates.TemplateResponse(
        request=request,
        name="prediction.html",
        context={
            "feature_names": feature_names,
            "prediction_feature_types": prediction_feature_types
        }
    )