from joblib import load
from sklearn.metrics import classification_report
from src.data_preprocessing import load_and_preprocess

def evaluate():
    model = load('model.joblib')
    _, X_test, _, y_test = load_and_preprocess('data/telco.csv')
    
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    evaluate()
