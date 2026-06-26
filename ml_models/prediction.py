def predict_value(model, values):
    prediction = model.predict( [values])[0]
    return prediction