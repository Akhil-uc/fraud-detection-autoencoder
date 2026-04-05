import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    X = df.drop('Class', axis=1)
    y = df['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def split_data(X, y):
    # Train only on normal data
    X_normal = X[y == 0]

    X_train, X_test, y_train, y_test = train_test_split(
        X_normal, y[y == 0], test_size=0.2, random_state=42
    )

    return X_train, X_test, y_test