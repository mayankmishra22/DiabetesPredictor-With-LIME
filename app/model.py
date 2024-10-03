import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle

class DiabetesModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
    def load_data(self):
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        data = pd.read_csv(url, names=self.columns + ['Outcome'])
        X = data.drop('Outcome', axis=1)
        y = data['Outcome']
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train_model(self):
        X_train, X_test, y_train, y_test = self.load_data()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Print metrics
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        
        return self.model
    
    def predict(self, input_data):
        return self.model.predict([input_data])

    def save_model(self, filepath='diabetes_model.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

# Ensure the script runs when executed
if __name__ == "__main__":
    model = DiabetesModel()
    model.train_model()