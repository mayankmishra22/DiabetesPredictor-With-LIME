import lime
import lime.lime_tabular
import numpy as np
import pandas as pd

class Explainer:
    def __init__(self, model):
        self.model = model
        self.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
    def lime_local_explanation(self, instance, X_train):
        # Create LIME explainer
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=self.columns,
            class_names=['No Diabetes', 'Diabetes'],
            mode='classification'
        )
        
        # Generate LIME explanation for the instance
        explanation = lime_explainer.explain_instance(
            data_row=instance.values[0],
            predict_fn=self.model.predict_proba
        )
        
        # Return the LIME explanation as a list of feature importances
        return explanation.as_list()
