"""
Advanced Polypharmacy Prediction Models
=====================================

This module contains advanced machine learning models specifically designed 
for polypharmacy side effect prediction, including ensemble methods and 
deep learning approaches.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedPolypharmacyPredictor:
    """
    Advanced prediction models for polypharmacy side effects
    """
    
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.encoders = {}
        self.scalers = {}
        self.models = {}
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare and encode data for advanced modeling"""
        print("Preparing data for advanced modeling...")
        
        # Create encoders
        self.encoders['stitch1'] = LabelEncoder()
        self.encoders['stitch2'] = LabelEncoder()
        self.encoders['side_effect'] = LabelEncoder()
        
        # Encode features
        self.data['STITCH_1_encoded'] = self.encoders['stitch1'].fit_transform(self.data['STITCH_1'])
        self.data['STITCH_2_encoded'] = self.encoders['stitch2'].fit_transform(self.data['STITCH_2'])
        
        # Create additional features
        self.data['drug_interaction_score'] = (
            self.data['STITCH_1_encoded'] * self.data['STITCH_2_encoded']
        ) % 1000
        
        self.data['drug_sum'] = self.data['STITCH_1_encoded'] + self.data['STITCH_2_encoded']
        self.data['drug_diff'] = abs(self.data['STITCH_1_encoded'] - self.data['STITCH_2_encoded'])
        
        # Prepare feature matrix
        self.X = self.data[['STITCH_1_encoded', 'STITCH_2_encoded', 
                           'drug_interaction_score', 'drug_sum', 'drug_diff']]
        
        # Prepare targets
        self.y_side_effect = self.encoders['side_effect'].fit_transform(
            self.data['Polypharmacy_Side_Effect']
        )
        
        # Scale features
        self.scalers['features'] = StandardScaler()
        self.X_scaled = self.scalers['features'].fit_transform(self.X)
        
        print(f"Prepared {self.X.shape[0]} samples with {self.X.shape[1]} features")
        print(f"Target has {len(np.unique(self.y_side_effect))} unique classes")
    
    def build_ensemble_model(self):
        """Build ensemble model combining multiple algorithms"""
        print("Building ensemble model...")
        
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        # Base models
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        lr = LogisticRegression(random_state=42, max_iter=1000)
        svm = SVC(probability=True, random_state=42)
        
        # Voting classifier
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('lr', lr),
                ('svm', svm)
            ],
            voting='soft'
        )
        
        # Stacking classifier
        stacking_clf = StackingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('lr', lr)
            ],
            final_estimator=LogisticRegression(),
            cv=5
        )
        
        self.models['voting'] = voting_clf
        self.models['stacking'] = stacking_clf
        
        return voting_clf, stacking_clf
    
    def build_neural_network(self):
        """Build neural network model"""
        print("Building neural network model...")
        
        # Neural network with sklearn
        nn_clf = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            random_state=42,
            max_iter=1000
        )
        
        self.models['neural_network'] = nn_clf
        
        return nn_clf
    
    def build_deep_learning_model(self):
        """Build deep learning model with TensorFlow"""
        print("Building deep learning model...")
        
        # Define model architecture
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.X_scaled.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(len(np.unique(self.y_side_effect)), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['deep_learning'] = model
        
        return model
    
    def train_all_models(self):
        """Train all models and evaluate performance"""
        print("Training all models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y_side_effect, test_size=0.2, 
            random_state=42, stratify=self.y_side_effect
        )
        
        results = {}
        
        # Build and train ensemble models
        voting_clf, stacking_clf = self.build_ensemble_model()
        
        print("Training Voting Classifier...")
        voting_clf.fit(X_train, y_train)
        voting_score = voting_clf.score(X_test, y_test)
        results['voting'] = voting_score
        print(f"Voting Classifier Accuracy: {voting_score:.3f}")
        
        print("Training Stacking Classifier...")
        stacking_clf.fit(X_train, y_train)
        stacking_score = stacking_clf.score(X_test, y_test)
        results['stacking'] = stacking_score
        print(f"Stacking Classifier Accuracy: {stacking_score:.3f}")
        
        # Train neural network
        nn_clf = self.build_neural_network()
        print("Training Neural Network...")
        nn_clf.fit(X_train, y_train)
        nn_score = nn_clf.score(X_test, y_test)
        results['neural_network'] = nn_score
        print(f"Neural Network Accuracy: {nn_score:.3f}")
        
        # Train deep learning model
        dl_model = self.build_deep_learning_model()
        print("Training Deep Learning Model...")
        
        # Convert to categorical for deep learning
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=len(np.unique(self.y_side_effect)))
        y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=len(np.unique(self.y_side_effect)))
        
        history = dl_model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        dl_score = dl_model.evaluate(X_test, y_test, verbose=0)[1]
        results['deep_learning'] = dl_score
        print(f"Deep Learning Model Accuracy: {dl_score:.3f}")
        
        # Store test data for later use
        self.X_test = X_test
        self.y_test = y_test
        
        return results
    
    def predict_drug_interaction(self, stitch_1, stitch_2, model_type='voting'):
        """Predict side effects for a specific drug interaction"""
        print(f"Predicting interaction between {stitch_1} and {stitch_2}...")
        
        try:
            # Encode the drugs
            stitch_1_enc = self.encoders['stitch1'].transform([stitch_1])[0]
            stitch_2_enc = self.encoders['stitch2'].transform([stitch_2])[0]
            
            # Create feature vector
            features = np.array([[
                stitch_1_enc,
                stitch_2_enc,
                (stitch_1_enc * stitch_2_enc) % 1000,
                stitch_1_enc + stitch_2_enc,
                abs(stitch_1_enc - stitch_2_enc)
            ]])
            
            # Scale features
            features_scaled = self.scalers['features'].transform(features)
            
            # Make prediction
            model = self.models[model_type]
            
            if model_type == 'deep_learning':
                prediction_proba = model.predict(features_scaled, verbose=0)
                prediction = np.argmax(prediction_proba, axis=1)[0]
                confidence = np.max(prediction_proba)
            else:
                prediction = model.predict(features_scaled)[0]
                if hasattr(model, 'predict_proba'):
                    prediction_proba = model.predict_proba(features_scaled)[0]
                    confidence = np.max(prediction_proba)
                else:
                    confidence = None
            
            # Decode prediction
            side_effect_code = self.encoders['side_effect'].inverse_transform([prediction])[0]
            
            # Find corresponding side effect name
            side_effect_name = self.data[
                self.data['Polypharmacy_Side_Effect'] == side_effect_code
            ]['Side_Effect_Name'].iloc[0] if len(self.data[
                self.data['Polypharmacy_Side_Effect'] == side_effect_code
            ]) > 0 else "Unknown"
            
            result = {
                'side_effect_code': side_effect_code,
                'side_effect_name': side_effect_name,
                'confidence': confidence,
                'model_used': model_type
            }
            
            print(f"Predicted side effect: {side_effect_name}")
            print(f"Side effect code: {side_effect_code}")
            if confidence:
                print(f"Confidence: {confidence:.3f}")
            
            return result
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
    
    def get_model_comparison(self):
        """Compare performance of all models"""
        results = self.train_all_models()
        
        print("\n=== MODEL COMPARISON ===")
        print("Model Performance (Accuracy):")
        for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model_name}: {accuracy:.3f}")
        
        return results
    
    def save_models(self):
        """Save trained models and encoders"""
        print("Saving models and encoders...")
        
        # Save sklearn models
        for name, model in self.models.items():
            if name != 'deep_learning':
                joblib.dump(model, f'e:/polyphormacy/{name}_model.pkl')
        
        # Save deep learning model
        if 'deep_learning' in self.models:
            self.models['deep_learning'].save('e:/polyphormacy/deep_learning_model.h5')
        
        # Save encoders and scalers
        joblib.dump(self.encoders, 'e:/polyphormacy/encoders.pkl')
        joblib.dump(self.scalers, 'e:/polyphormacy/scalers.pkl')
        
        print("Models saved successfully!")
    
    def load_models(self):
        """Load pre-trained models"""
        print("Loading pre-trained models...")
        
        try:
            # Load sklearn models
            model_names = ['voting', 'stacking', 'neural_network']
            for name in model_names:
                try:
                    self.models[name] = joblib.load(f'e:/polyphormacy/{name}_model.pkl')
                except:
                    print(f"Could not load {name} model")
            
            # Load deep learning model
            try:
                self.models['deep_learning'] = keras.models.load_model('e:/polyphormacy/deep_learning_model.h5')
            except:
                print("Could not load deep learning model")
            
            # Load encoders and scalers
            self.encoders = joblib.load('e:/polyphormacy/encoders.pkl')
            self.scalers = joblib.load('e:/polyphormacy/scalers.pkl')
            
            print("Models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")

def main():
    """Main function for advanced modeling"""
    print("ADVANCED POLYPHARMACY PREDICTION MODELS")
    print("=" * 50)
    
    # Initialize predictor
    predictor = AdvancedPolypharmacyPredictor('e:/polyphormacy/polypharmacy_data.csv')
    
    # Train and compare models
    results = predictor.get_model_comparison()
    
    # Save models
    predictor.save_models()
    
    # Example predictions with different models
    print("\n=== EXAMPLE PREDICTIONS ===")
    drug_pairs = [
        ('CID000002173', 'CID000003345'),
    ]
    
    for stitch_1, stitch_2 in drug_pairs:
        print(f"\nDrug pair: {stitch_1} + {stitch_2}")
        for model_type in ['voting', 'stacking', 'neural_network']:
            if model_type in predictor.models:
                print(f"\n{model_type.upper()} MODEL:")
                predictor.predict_drug_interaction(stitch_1, stitch_2, model_type)
    
    print("\n" + "=" * 50)
    print("ADVANCED MODELING COMPLETE!")

if __name__ == "__main__":
    main()
