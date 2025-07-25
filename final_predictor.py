"""
Final Working Polypharmacy Prediction System
============================================

A working version with proper handling of class imbalance and data issues.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FinalPolypharmacyPredictor:
    """
    Final working polypharmacy prediction system
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
        
        self.data['STITCH_1_encoded'] = self.encoders['stitch1'].fit_transform(self.data['STITCH_1'])
        self.data['STITCH_2_encoded'] = self.encoders['stitch2'].fit_transform(self.data['STITCH_2'])
        
        # Create interaction features
        self.data['drug_interaction_score'] = (self.data['STITCH_1_encoded'] * self.data['STITCH_2_encoded']) % 10000
        self.data['drug_sum'] = self.data['STITCH_1_encoded'] + self.data['STITCH_2_encoded']
        self.data['drug_diff'] = abs(self.data['STITCH_1_encoded'] - self.data['STITCH_2_encoded'])
        
        # Add severity and system mappings
        severity_mapping = self._get_severity_mapping()
        system_mapping = self._get_system_mapping()
        
        self.data['severity'] = self.data['Side_Effect_Name'].map(severity_mapping).fillna('moderate')
        self.data['affected_system'] = self.data['Side_Effect_Name'].map(system_mapping).fillna('other')
        
        # For prediction targets, we'll use simplified categories
        # Group similar side effects together to reduce class imbalance
        self.data['side_effect_category'] = self.data['Side_Effect_Name'].apply(self._categorize_side_effect)
        
        # Encode the simplified targets
        self.encoders['severity'] = LabelEncoder()
        self.encoders['system'] = LabelEncoder()
        self.encoders['category'] = LabelEncoder()
        
        self.data['severity_encoded'] = self.encoders['severity'].fit_transform(self.data['severity'])
        self.data['system_encoded'] = self.encoders['system'].fit_transform(self.data['affected_system'])
        self.data['category_encoded'] = self.encoders['category'].fit_transform(self.data['side_effect_category'])
        
        print(f"Feature engineering complete. Dataset shape: {self.data.shape}")
        print(f"Severity classes: {self.data['severity'].value_counts().to_dict()}")
        print(f"System classes: {len(self.data['affected_system'].unique())}")
        print(f"Category classes: {len(self.data['side_effect_category'].unique())}")
        
    def _categorize_side_effect(self, side_effect):
        """Categorize side effects into broader groups"""
        side_effect_lower = side_effect.lower()
        
        # Cardiovascular
        if any(word in side_effect_lower for word in ['heart', 'cardiac', 'blood pressure', 'angina', 'bradycardia', 'chest pain']):
            return 'cardiovascular'
        
        # Respiratory
        if any(word in side_effect_lower for word in ['lung', 'respiratory', 'breathing', 'pneumonia', 'apnea']):
            return 'respiratory'
        
        # Neurological
        if any(word in side_effect_lower for word in ['brain', 'neurological', 'headache', 'dizziness', 'confusion']):
            return 'neurological'
        
        # Gastrointestinal
        if any(word in side_effect_lower for word in ['abdominal', 'stomach', 'nausea', 'diarrhea', 'constipat']):
            return 'gastrointestinal'
        
        # Metabolic
        if any(word in side_effect_lower for word in ['diabetes', 'glucose', 'metabolic', 'magnesium']):
            return 'metabolic'
        
        # Blood-related
        if any(word in side_effect_lower for word in ['blood', 'anemia', 'bleeding', 'thrombocyt']):
            return 'hematological'
        
        # Pain-related
        if any(word in side_effect_lower for word in ['pain', 'ache', 'aching']):
            return 'pain'
        
        # Infectious
        if any(word in side_effect_lower for word in ['infection', 'sepsis', 'fever']):
            return 'infectious'
        
        return 'other'
    
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
            'loss of consciousness': 'severe', 'coughing blood': 'severe', 'lymphoma': 'severe',
            
            # Moderate - default for most side effects
            # Mild
            'flatulence': 'mild', 'nausea': 'mild', 'diarrhea': 'mild', 'constipated': 'mild',
            'dizziness': 'mild', 'Fatigue': 'mild', 'drowsiness': 'mild',
            'weight gain': 'mild', 'loss of weight': 'mild', 'bruise': 'mild',
            'abdominal distension': 'mild', 'chill': 'mild', 'flu': 'mild'
        }
    
    def _get_system_mapping(self):
        """Get body system mapping for side effects"""
        return {
            # Use the categorization function for consistency
        }
    
    def train_models(self):
        """Train classification models with proper handling of class imbalance"""
        print("\n=== TRAINING MODELS ===")
        
        # Prepare features
        feature_cols = ['STITCH_1_encoded', 'STITCH_2_encoded', 'drug_interaction_score', 'drug_sum', 'drug_diff']
        X = self.data[feature_cols]
        
        # Define targets with their minimum class requirements
        targets = {
            'severity': ('severity_encoded', 2),
            'system': ('category_encoded', 5),  # Use categorized side effects
        }
        
        results = {}
        
        for target_name, (target_col, min_samples) in targets.items():
            print(f"\nTraining model for {target_name} prediction...")
            
            y = self.data[target_col]
            
            # Check class distribution and filter classes with too few samples
            class_counts = y.value_counts()
            valid_classes = class_counts[class_counts >= min_samples].index
            
            if len(valid_classes) < 2:
                print(f"  Skipping {target_name}: insufficient classes with minimum samples")
                continue
            
            # Filter data to only include valid classes
            valid_mask = y.isin(valid_classes)
            X_filtered = X[valid_mask]
            y_filtered = y[valid_mask]
            
            print(f"  Using {len(valid_classes)} classes with {len(X_filtered)} samples")
            
            try:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
                )
                
                # Train Random Forest model
                model = RandomForestClassifier(
                    n_estimators=100, 
                    random_state=42, 
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight='balanced'  # Handle class imbalance
                )
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                results[target_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'classes': len(valid_classes),
                    'samples': len(X_filtered)
                }
                
                self.models[target_name] = model
                
                print(f"  Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
                print(f"  Classes: {len(valid_classes)}, Samples: {len(X_filtered)}")
                
            except Exception as e:
                print(f"  Error training {target_name} model: {str(e)}")
        
        self.results = results
        return results
    
    def create_binary_classifier(self):
        """Create a binary classifier for interaction prediction"""
        print("\n=== CREATING BINARY INTERACTION CLASSIFIER ===")
        
        # For binary classification, we'll predict if a drug pair causes severe side effects
        self.data['has_severe_interaction'] = (self.data['severity'] == 'severe').astype(int)
        
        # Features
        feature_cols = ['STITCH_1_encoded', 'STITCH_2_encoded', 'drug_interaction_score', 'drug_sum', 'drug_diff']
        X = self.data[feature_cols]
        y = self.data['has_severe_interaction']
        
        print(f"Binary target distribution: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        self.models['binary'] = model
        self.results['binary'] = {
            'accuracy': accuracy,
            'f1_score': f1
        }
        
        print(f"Binary classifier - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
    
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
                abs(drug1_encoded - drug2_encoded)
            ]])
            
            predictions = {}
            
            # Binary prediction
            if 'binary' in self.models:
                binary_pred = self.models['binary'].predict(features)[0]
                binary_proba = self.models['binary'].predict_proba(features)[0]
                predictions['has_severe_interaction'] = {
                    'prediction': bool(binary_pred),
                    'confidence': float(binary_proba.max())
                }
            
            # Other predictions
            for target in ['severity', 'system']:
                if target in self.models:
                    model = self.models[target]
                    pred = model.predict(features)[0]
                    proba = model.predict_proba(features)[0]
                    
                    if target == 'severity':
                        decoded = self.encoders['severity'].inverse_transform([pred])[0]
                    else:
                        decoded = self.encoders['category'].inverse_transform([pred])[0]
                    
                    predictions[target] = {
                        'prediction': decoded,
                        'confidence': float(proba.max())
                    }
            
            # Check for known interactions
            known_interactions = self.data[
                ((self.data['STITCH_1'] == drug1) & (self.data['STITCH_2'] == drug2)) |
                ((self.data['STITCH_1'] == drug2) & (self.data['STITCH_2'] == drug1))
            ]
            
            result = {
                'drug1': drug1,
                'drug2': drug2,
                'predictions': predictions,
                'known_interactions': known_interactions[['Side_Effect_Name', 'severity', 'side_effect_category']].to_dict('records'),
                'has_known_interactions': len(known_interactions) > 0,
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {'error': f'Prediction error: {str(e)}'}
    
    def save_models_and_data(self):
        """Save all models and processed data"""
        print("\nSaving models and data...")
        
        # Save encoders
        joblib.dump(self.encoders, 'e:/polyphormacy/encoders.pkl')
        
        # Save models
        for model_name, model in self.models.items():
            filename = f"e:/polyphormacy/{model_name}_model.pkl"
            joblib.dump(model, filename)
        
        # Save results summary
        results_summary = {}
        for target, result in self.results.items():
            if isinstance(result, dict) and 'model' in result:
                results_summary[target] = {k: v for k, v in result.items() if k != 'model'}
            else:
                results_summary[target] = result
        
        with open('e:/polyphormacy/model_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Save processed data
        self.data.to_csv('e:/polyphormacy/processed_polypharmacy_data.csv', index=False)
        
        print("Models and data saved successfully!")

def main():
    """Main training pipeline"""
    print("FINAL POLYPHARMACY PREDICTION SYSTEM")
    print("=" * 60)
    
    # Initialize predictor
    predictor = FinalPolypharmacyPredictor('e:/polyphormacy/polypharmacy_data.csv')
    
    # Load and preprocess data
    predictor.load_and_clean_data()
    predictor.feature_engineering()
    
    # Train models
    predictor.train_models()
    predictor.create_binary_classifier()
    
    # Save everything
    predictor.save_models_and_data()
    
    # Example prediction
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTION")
    
    # Get two drugs from the dataset for testing
    drug1 = predictor.data['STITCH_1'].iloc[0]
    drug2 = predictor.data['STITCH_2'].iloc[0]
    
    result = predictor.predict_interaction(drug1, drug2)
    print(f"Prediction for {drug1} + {drug2}:")
    if 'error' not in result:
        print(f"  Predictions: {result['predictions']}")
        print(f"  Has known interactions: {result['has_known_interactions']}")
        if result['has_known_interactions']:
            print(f"  Known interactions: {len(result['known_interactions'])}")
    else:
        print(f"  Error: {result['error']}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("Ready to start the API server!")

if __name__ == "__main__":
    main()
