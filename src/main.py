from src.preprocessing import load_data, preprocess, split_data
from src.model import build_model, train, predict
from src.evaluation import evaluate, plot_scores

def main():
    print("Loading data...")
    df = load_data('../data/fraud.csv')

    print("Preprocessing...")
    X, y = preprocess(df)

    print("Splitting data...")
    X_train, X_test, y_test = split_data(X, y)

    print("Building model...")
    model = build_model()

    print("Training on NORMAL transactions only...")
    model = train(model, X_train)

    print("Testing on full dataset...")
    y_pred, scores = predict(model, X)

    print("Evaluating...")
    evaluate(y, y_pred)

    print("Visualizing anomaly scores...")
    plot_scores(scores)

if __name__ == "__main__":
    main()