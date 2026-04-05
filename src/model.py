from pyod.models.auto_encoder import AutoEncoder

def build_model():
    model = AutoEncoder(
        contamination=0.02,
        epoch_num=20,
        batch_size=64,
        verbose=1
    )
    return model

def train(model, X_train):
    model.fit(X_train)
    return model

def predict(model, X):
    y_pred = model.predict(X)
    scores = model.decision_function(X)
    return y_pred, scores