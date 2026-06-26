def detect_problem(df,target):
    y=df[target]
    if y.dtype=="objects" or y.nunique()<=10:
        return "classification"
    else:
        return "regression"