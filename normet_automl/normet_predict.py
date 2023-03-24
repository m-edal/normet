#required package: from flaml import AutoML
def normet_predict(automl, df=None):
    x = automl.predict(df)
    return x
