"""
Simplified Polypharmacy Prediction System
=========================================

A working version focused on multiclass classification and existing data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimplifiedPolypharmacyPredictor:
    """
    Simplified polypharmacy prediction system
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.encoders = {}
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_and_clean_data(self):
        """Load and clean the polypharmacy data"""
        print("Loading and cleaning data...")
        
        # Read the CSV file
        self.data = pd.read_csv(self.data_path, sep='\t')
        
        # If tab separation doesn't work, try comma
        if len(self.data.columns) == 1:
            self.data = pd.read_csv(self.data_path)
        
        # Clean column names
        if self.data.columns[0].startswith('#'):
            self.data.columns = ['STITCH_1', 'STITCH_2', 'Polypharmacy_Side_Effect', 'Side_Effect_Name']
        
        # Remove any rows that are comments
        self.data = self.data[~self.data['STITCH_1'].astype(str).str.startswith('#')]
        
        # Clean whitespace
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                self.data[col] = self.data[col].astype(str).str.strip()
        
        # Remove duplicates
        self.data = self.data.drop_duplicates()
        
        print(f"Loaded {len(self.data)} drug interaction records")
        print(f"Unique drugs: {len(set(self.data['STITCH_1'].tolist() + self.data['STITCH_2'].tolist()))}")
        print(f"Unique side effects: {self.data['Polypharmacy_Side_Effect'].nunique()}")
        
        return self.data
    
    def feature_engineering(self):
        """Basic feature engineering"""
        print("Performing feature engineering...")
        
        # Encode categorical variables
        self.encoders['stitch1'] = LabelEncoder()
        self.encoders['stitch2'] = LabelEncoder()
        self.encoders['side_effect'] = LabelEncoder()
        self.encoders['side_effect_name'] = LabelEncoder()
        
        self.data['STITCH_1_encoded'] = self.encoders['stitch1'].fit_transform(self.data['STITCH_1'])
        self.data['STITCH_2_encoded'] = self.encoders['stitch2'].fit_transform(self.data['STITCH_2'])
        self.data['Side_Effect_encoded'] = self.encoders['side_effect'].fit_transform(self.data['Polypharmacy_Side_Effect'])
        self.data['Side_Effect_Name_encoded'] = self.encoders['side_effect_name'].fit_transform(self.data['Side_Effect_Name'])
        
        # Create interaction features
        self.data['drug_interaction_score'] = (self.data['STITCH_1_encoded'] * self.data['STITCH_2_encoded']) % 10000
        self.data['drug_sum'] = self.data['STITCH_1_encoded'] + self.data['STITCH_2_encoded']
        self.data['drug_diff'] = abs(self.data['STITCH_1_encoded'] - self.data['STITCH_2_encoded'])
        self.data['drug_ratio'] = np.where(self.data['STITCH_2_encoded'] != 0, 
                                          self.data['STITCH_1_encoded'] / self.data['STITCH_2_encoded'], 0)
        
        # Add severity and system mappings
        self.data['severity'] = self.data['Side_Effect_Name'].map(self._get_severity_mapping()).fillna('moderate')
        self.data['affected_system'] = self.data['Side_Effect_Name'].map(self._get_system_mapping()).fillna('other')
        
        # Encode severity and system
        self.encoders['severity'] = LabelEncoder()
        self.encoders['system'] = LabelEncoder()
        self.data['severity_encoded'] = self.encoders['severity'].fit_transform(self.data['severity'])
        self.data['system_encoded'] = self.encoders['system'].fit_transform(self.data['affected_system'])
        
        print(f"Feature engineering complete. Dataset shape: {self.data.shape}")
        
    def _get_severity_mapping(self):
        """Get severity mapping for side effects"""
        return {
            # Severe (life-threatening)
            'asystole': 'severe', 'cerebral infarct': 'severe', 'rupture of spleen': 'severe',
            'respiratory failure': 'severe', 'Acute Respiratory Distress Syndrome': 'severe',
            'sepsis': 'severe', 'hepatic necrosis': 'severe', 'heart attack': 'severe',
            'apoplexy': 'severe', 'Embolism pulmonary': 'severe', 'toxic shock': 'severe',
            'Hepatic failure': 'severe', 'convulsion': 'severe', 'deep vein thromboses': 'severe',
            'Bleeding': 'severe', 'thrombocytopenia': 'severe', 'Leukaemia': 'severe',
            'pneumonia': 'severe', 'neumonia': 'severe', 'lung edema': 'severe',
            'lung infiltration': 'severe', 'atelectasis': 'severe', 'peritonitis': 'severe',
            'loss of consciousness': 'severe', 'coughing blood': 'severe', 'lymphoma': 'severe',
            
            # Moderate
            'hypermagnesemia': 'moderate', 'hypoglycaemia': 'moderate', 'hyperglycaemia': 'moderate',
            'Drug hypersensitivity': 'moderate', 'allergies': 'moderate', 'bradycardia': 'moderate',
            'High blood pressure': 'moderate', 'chest pain': 'moderate', 'kidney failure': 'moderate',
            'Diabetes': 'moderate', 'angina': 'moderate', 'AFIB': 'moderate',
            'cardiac failure': 'moderate', 'abnormal movements': 'moderate', 'alkalosis': 'moderate',
            'Acidosis': 'moderate', 'pain in throat': 'moderate', 'Head ache': 'moderate',
            'Back Ache': 'moderate', 'arthritis': 'moderate', 'hypoxia': 'moderate',
            
            # Mild
            'flatulence': 'mild', 'nausea': 'mild', 'diarrhea': 'mild', 'constipated': 'mild',
            'dizziness': 'mild', 'Fatigue': 'mild', 'drowsiness': 'mild', 'bad breath': 'mild',
            'weight gain': 'mild', 'loss of weight': 'mild', 'bruise': 'mild',
            'abdominal distension': 'mild', 'chill': 'mild', 'flu': 'mild'
        }
    
    def _get_system_mapping(self):
        """Get body system mapping for side effects"""
        return {
            # Cardiovascular
            'asystole': 'cardiovascular', 'bradycardia': 'cardiovascular', 'chest pain': 'cardiovascular',
            'angina': 'cardiovascular', 'AFIB': 'cardiovascular', 'cardiac failure': 'cardiovascular',
            'heart attack': 'cardiovascular', 'High blood pressure': 'cardiovascular',
            'Cardiomyopathy': 'cardiovascular', 'Cardiac decompensation': 'cardiovascular',
            
            # Respiratory
            'respiratory failure': 'respiratory', 'Acute Respiratory Distress Syndrome': 'respiratory',
            'lung edema': 'respiratory', 'lung infiltration': 'respiratory', 'atelectasis': 'respiratory',
            'Difficulty breathing': 'respiratory', 'pneumonia': 'respiratory', 'neumonia': 'respiratory',
            'Apnea': 'respiratory', 'bronchitis': 'respiratory', 'hypoxia': 'respiratory',
            'coughing blood': 'respiratory',
            
            # Neurological
            'cerebral infarct': 'neurological', 'loss of consciousness': 'neurological',
            'abnormal movements': 'neurological', 'Head ache': 'neurological', 'dizziness': 'neurological',
            'drowsiness': 'neurological', 'convulsion': 'neurological', 'Hallucination': 'neurological',
            'confusion': 'neurological', 'Anxiety': 'neurological',
            
            # Gastrointestinal
            'flatulence': 'gastrointestinal', 'nausea': 'gastrointestinal', 'diarrhea': 'gastrointestinal',
            'constipated': 'gastrointestinal', 'abdominal distension': 'gastrointestinal',
            'peritonitis': 'gastrointestinal', 'Gastrointestinal Obstruction': 'gastrointestinal',
            'acid reflux': 'gastrointestinal', 'emesis': 'gastrointestinal',
            
            # Metabolic
            'hypermagnesemia': 'metabolic', 'hypoglycaemia': 'metabolic', 'hyperglycaemia': 'metabolic',
            'alkalosis': 'metabolic', 'Acidosis': 'metabolic', 'Diabetes': 'metabolic',
            'Blood calcium decreased': 'metabolic', 'Hypomagnesaemia': 'metabolic',
            
            # Hematological
            'Bleeding': 'hematological', 'thrombocytopenia': 'hematological', 'Leukaemia': 'hematological',
            'anaemia': 'hematological', 'deep vein thromboses': 'hematological',
            'increased white blood cell count': 'hematological', 'leucocytosis': 'hematological',
            
            # Immunological
            'Drug hypersensitivity': 'immunological', 'allergies': 'immunological',
            'allergic dermatitis': 'immunological',
            
            # Other systems
            'hepatic necrosis': 'hepatic', 'Hepatic failure': 'hepatic',
            'kidney failure': 'renal', 'sepsis': 'systemic',
            'Back Ache': 'musculoskeletal'
        }
    
    def train_models(self):
        """Train multiple classification models"""
        print("\n=== TRAINING MODELS ===")
        
        # Prepare feature sets
        basic_features = ['STITCH_1_encoded', 'STITCH_2_encoded']
        interaction_features = basic_features + ['drug_interaction_score', 'drug_sum', 'drug_diff', 'drug_ratio']
        
        # Define targets
        targets = {
            'side_effect': 'Side_Effect_encoded',
            'severity': 'severity_encoded', 
            'system': 'system_encoded'
        }
        
        # Initialize models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=20),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, max_depth=10),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}
        
        for target_name, target_col in targets.items():
            print(f"\nTraining models for {target_name} prediction...")
            
            # Use interaction features for better performance
            X = self.data[interaction_features]
            y = self.data[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features for logistic regression
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            target_results = {}
            
            for model_name, model in models.items():
                print(f"  Training {model_name}...")
                
                try:
                    if model_name == 'Logistic Regression':
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    # Cross-validation
                    if model_name == 'Logistic Regression':
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1_weighted')
                    else:
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
                    
                    target_results[model_name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'feature_set': 'interaction_features'
                    }
                    
                    print(f"    Accuracy: {accuracy:.3f}, F1: {f1:.3f}, CV: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                    
                except Exception as e:
                    print(f"    Error training {model_name}: {str(e)}")
            
            results[target_name] = target_results
        
        self.results = results
        self.models = {}
        
        # Store best models
        for target_name, target_results in results.items():
            best_model_name = max(target_results.keys(), key=lambda x: target_results[x]['f1_score'])
            self.models[f'best_{target_name}'] = target_results[best_model_name]['model']
            print(f"\nBest model for {target_name}: {best_model_name} (F1: {target_results[best_model_name]['f1_score']:.3f})")
        
        return results
    
    def predict_interaction(self, drug1, drug2):
        """Predict interaction between two drugs"""
        try:
            # Check if drugs exist in our encoders
            if drug1 not in self.encoders['stitch1'].classes_:
                return {'error': f'Drug {drug1} not found in database'}
            
            if drug2 not in self.encoders['stitch2'].classes_:
                return {'error': f'Drug {drug2} not found in database'}
            
            # Encode drugs
            drug1_encoded = self.encoders['stitch1'].transform([drug1])[0]
            drug2_encoded = self.encoders['stitch2'].transform([drug2])[0]
            
            # Create feature vector
            features = np.array([[
                drug1_encoded,
                drug2_encoded,
                (drug1_encoded * drug2_encoded) % 10000,
                drug1_encoded + drug2_encoded,
                abs(drug1_encoded - drug2_encoded),
                drug1_encoded / drug2_encoded if drug2_encoded != 0 else 0
            ]])
            
            predictions = {}
            
            # Make predictions for each target
            for target in ['side_effect', 'severity', 'system']:
                model = self.models[f'best_{target}']
                
                pred = model.predict(features)[0]
                proba = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else None
                
                # Decode prediction
                if target == 'side_effect':
                    decoded = self.encoders['side_effect'].inverse_transform([pred])[0]
                    # Find corresponding side effect name
                    side_effect_name = self.data[self.data['Polypharmacy_Side_Effect'] == decoded]['Side_Effect_Name'].iloc[0]
                    predictions[target] = {'code': decoded, 'name': side_effect_name}
                elif target == 'severity':
                    predictions[target] = self.encoders['severity'].inverse_transform([pred])[0]
                elif target == 'system':
                    predictions[target] = self.encoders['system'].inverse_transform([pred])[0]
                
                if proba is not None:
                    predictions[f'{target}_confidence'] = float(proba.max())
            
            # Check for known interactions
            known_interactions = self.data[
                ((self.data['STITCH_1'] == drug1) & (self.data['STITCH_2'] == drug2)) |
                ((self.data['STITCH_1'] == drug2) & (self.data['STITCH_2'] == drug1))
            ]
            
            result = {
                'drug1': drug1,
                'drug2': drug2,
                'predictions': predictions,
                'known_interactions': known_interactions[['Side_Effect_Name', 'severity', 'affected_system']].to_dict('records'),
                'has_known_interactions': len(known_interactions) > 0,
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {'error': f'Prediction error: {str(e)}'}
    
    def save_models_and_data(self):
        """Save all models and processed data"""
        print("\nSaving models and data...")
        
        # Save encoders and scaler
        joblib.dump(self.encoders, 'e:/polyphormacy/encoders.pkl')
        joblib.dump(self.scaler, 'e:/polyphormacy/scaler.pkl')
        
        # Save models
        for model_name, model in self.models.items():
            filename = f"e:/polyphormacy/{model_name.replace(' ', '_').lower()}.pkl"
            joblib.dump(model, filename)
        
        # Save results summary
        results_summary = {}
        for target, models in self.results.items():
            results_summary[target] = {
                model_name: {
                    'accuracy': result['accuracy'],
                    'f1_score': result['f1_score'],
                    'cv_mean': result['cv_mean'],
                    'cv_std': result['cv_std']
                }
                for model_name, result in models.items()
            }
        
        with open('e:/polyphormacy/model_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Save processed data
        self.data.to_csv('e:/polyphormacy/processed_polypharmacy_data.csv', index=False)
        
        print("Models and data saved successfully!")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n=== POLYPHARMACY ANALYSIS SUMMARY ===")
        
        # Data summary
        print(f"Dataset Overview:")
        print(f"  Total drug interactions: {len(self.data)}")
        print(f"  Unique drugs: {len(set(self.data['STITCH_1'].tolist() + self.data['STITCH_2'].tolist()))}")
        print(f"  Unique side effects: {self.data['Polypharmacy_Side_Effect'].nunique()}")
        print(f"  Unique drug pairs: {len(self.data[['STITCH_1', 'STITCH_2']].drop_duplicates())}")
        
        # Severity distribution
        print(f"\nSeverity Distribution:")
        severity_counts = self.data['severity'].value_counts()
        for severity, count in severity_counts.items():
            print(f"  {severity}: {count} ({count/len(self.data)*100:.1f}%)")
        
        # System distribution
        print(f"\nAffected Systems:")
        system_counts = self.data['affected_system'].value_counts()
        for system, count in system_counts.head(5).items():
            print(f"  {system}: {count} ({count/len(self.data)*100:.1f}%)")
        
        # Model performance
        print(f"\nBest Model Performance:")
        for target, models in self.results.items():
            best_model = max(models.keys(), key=lambda x: models[x]['f1_score'])
            best_f1 = models[best_model]['f1_score']
            print(f"  {target}: {best_model} (F1: {best_f1:.3f})")

def main():
    """Main training pipeline"""
    print("SIMPLIFIED POLYPHARMACY PREDICTION SYSTEM")
    print("=" * 60)
    
    # Initialize predictor
    predictor = SimplifiedPolypharmacyPredictor('e:/polyphormacy/polypharmacy_data.csv')
    
    # Load and preprocess data
    predictor.load_and_clean_data()
    predictor.feature_engineering()
    
    # Train models
    predictor.train_models()
    
    # Save everything
    predictor.save_models_and_data()
    
    # Generate summary
    predictor.generate_summary_report()
    
    # Example prediction
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTION")
    
    # Get two drugs from the dataset for testing
    drug1 = predictor.data['STITCH_1'].iloc[0]
    drug2 = predictor.data['STITCH_2'].iloc[0]
    
    result = predictor.predict_interaction(drug1, drug2)
    print(f"Prediction for {drug1} + {drug2}:")
    if 'error' not in result:
        print(f"  Predicted side effect: {result['predictions']['side_effect']['name']}")
        print(f"  Predicted severity: {result['predictions']['severity']}")
        print(f"  Predicted system: {result['predictions']['system']}")
        print(f"  Has known interactions: {result['has_known_interactions']}")
    else:
        print(f"  Error: {result['error']}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("Ready to start the API server!")

if __name__ == "__main__":
    main()
