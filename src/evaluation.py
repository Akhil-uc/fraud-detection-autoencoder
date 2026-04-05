from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def evaluate(y_true, y_pred):
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

def plot_scores(scores):
    plt.hist(scores, bins=50)
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.show()