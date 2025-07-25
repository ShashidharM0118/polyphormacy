"""
Advanced Polypharmacy Prediction System
======================================

This module implements comprehensive data preprocessing, feature engineering,
and multiple machine learning models for polypharmacy side effect prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import networkx as nx
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedPolypharmacyPredictor:
    """
    Advanced polypharmacy prediction system with comprehensive preprocessing
    and multiple model architectures.
    """
    
    def __init__(self, data_path):
        """Initialize the predictor with data"""
        self.data_path = data_path
        self.data = None
        self.encoders = {}
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.drug_graph = None
        self.feature_importance = {}
        
    def load_and_clean_data(self):
        """Load and clean the polypharmacy data"""
        print("Loading and cleaning data...")
        
        # Handle the header with spaces and special characters
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
        print(f"Unique drug pairs: {len(self.data[['STITCH_1', 'STITCH_2']].drop_duplicates())}")
        print(f"Unique side effects: {self.data['Polypharmacy_Side_Effect'].nunique()}")
        
        return self.data
    
    def comprehensive_feature_engineering(self):
        """Comprehensive feature engineering for drug interactions"""
        print("Performing comprehensive feature engineering...")
        
        # Step 1: Basic encoding
        self.encoders['stitch1'] = LabelEncoder()
        self.encoders['stitch2'] = LabelEncoder()
        self.encoders['side_effect'] = LabelEncoder()
        self.encoders['side_effect_name'] = LabelEncoder()
        
        self.data['STITCH_1_encoded'] = self.encoders['stitch1'].fit_transform(self.data['STITCH_1'])
        self.data['STITCH_2_encoded'] = self.encoders['stitch2'].fit_transform(self.data['STITCH_2'])
        self.data['Side_Effect_encoded'] = self.encoders['side_effect'].fit_transform(self.data['Polypharmacy_Side_Effect'])
        self.data['Side_Effect_Name_encoded'] = self.encoders['side_effect_name'].fit_transform(self.data['Side_Effect_Name'])
        
        # Step 2: Interaction features
        self.data['drug_pair_hash'] = (self.data['STITCH_1_encoded'] * 1000 + self.data['STITCH_2_encoded']).astype(str)
        self.data['drug_interaction_score'] = (self.data['STITCH_1_encoded'] * self.data['STITCH_2_encoded']) % 10000
        self.data['drug_sum'] = self.data['STITCH_1_encoded'] + self.data['STITCH_2_encoded']
        self.data['drug_diff'] = abs(self.data['STITCH_1_encoded'] - self.data['STITCH_2_encoded'])
        self.data['drug_ratio'] = np.where(self.data['STITCH_2_encoded'] != 0, 
                                          self.data['STITCH_1_encoded'] / self.data['STITCH_2_encoded'], 0)
        
        # Step 3: Drug-specific features
        drug1_stats = self.data.groupby('STITCH_1_encoded').agg({
            'Side_Effect_encoded': ['count', 'nunique'],
            'STITCH_2_encoded': 'nunique'
        }).reset_index()
        drug1_stats.columns = ['STITCH_1_encoded', 'drug1_total_interactions', 'drug1_unique_effects', 'drug1_partners']
        
        drug2_stats = self.data.groupby('STITCH_2_encoded').agg({
            'Side_Effect_encoded': ['count', 'nunique'],
            'STITCH_1_encoded': 'nunique'
        }).reset_index()
        drug2_stats.columns = ['STITCH_2_encoded', 'drug2_total_interactions', 'drug2_unique_effects', 'drug2_partners']
        
        self.data = self.data.merge(drug1_stats, on='STITCH_1_encoded', how='left')
        self.data = self.data.merge(drug2_stats, on='STITCH_2_encoded', how='left')
        
        # Step 4: Side effect classification features
        severity_mapping = self._create_severity_mapping()
        system_mapping = self._create_system_mapping()
        
        self.data['severity'] = self.data['Side_Effect_Name'].map(severity_mapping).fillna('moderate')
        self.data['affected_system'] = self.data['Side_Effect_Name'].map(system_mapping).fillna('other')
        
        # Encode severity and system
        self.encoders['severity'] = LabelEncoder()
        self.encoders['system'] = LabelEncoder()
        self.data['severity_encoded'] = self.encoders['severity'].fit_transform(self.data['severity'])
        self.data['system_encoded'] = self.encoders['system'].fit_transform(self.data['affected_system'])
        
        # Step 5: Graph-based features
        self._create_drug_interaction_graph()
        self._add_graph_features()
        
        print(f"Feature engineering complete. Dataset shape: {self.data.shape}")
        
        return self.data
    
    def _create_severity_mapping(self):
        """Create severity mapping for side effects"""
        return {
            # Severe (life-threatening)
            'asystole': 'severe', 'cerebral infarct': 'severe', 'rupture of spleen': 'severe',
            'respiratory failure': 'severe', 'Acute Respiratory Distress Syndrome': 'severe',
            'sepsis': 'severe', 'hepatic necrosis': 'severe', 'attempted suicide': 'severe',
            'loss of consciousness': 'severe', 'heart attack': 'severe', 'apoplexy': 'severe',
            'Embolism pulmonary': 'severe', 'toxic shock': 'severe', 'Hepatic failure': 'severe',
            'convulsion': 'severe', 'deep vein thromboses': 'severe', 'Bleeding': 'severe',
            'thrombocytopenia': 'severe', 'Leukaemia': 'severe', 'lymphoma': 'severe',
            'pneumonia': 'severe', 'neumonia': 'severe', 'lung edema': 'severe',
            'lung infiltration': 'severe', 'atelectasis': 'severe', 'peritonitis': 'severe',
            
            # Moderate
            'hypermagnesemia': 'moderate', 'hypoglycaemia': 'moderate', 'hyperglycaemia': 'moderate',
            'Drug hypersensitivity': 'moderate', 'allergies': 'moderate', 'bradycardia': 'moderate',
            'High blood pressure': 'moderate', 'chest pain': 'moderate', 'kidney failure': 'moderate',
            'Diabetes': 'moderate', 'angina': 'moderate', 'AFIB': 'moderate',
            'cardiac failure': 'moderate', 'abnormal movements': 'moderate', 'alkalosis': 'moderate',
            'Acidosis': 'moderate', 'pain in throat': 'moderate', 'Head ache': 'moderate',
            'back pain': 'moderate', 'Back Ache': 'moderate', 'arthritis': 'moderate',
            
            # Mild
            'flatulence': 'mild', 'nausea': 'mild', 'diarrhea': 'mild', 'constipated': 'mild',
            'dizziness': 'mild', 'Fatigue': 'mild', 'drowsiness': 'mild', 'bad breath': 'mild',
            'weight gain': 'mild', 'loss of weight': 'mild', 'bruise': 'mild',
            'abdominal distension': 'mild', 'chill': 'mild', 'flu': 'mild'
        }
    
    def _create_system_mapping(self):
        """Create body system mapping for side effects"""
        return {
            # Cardiovascular
            'asystole': 'cardiovascular', 'bradycardia': 'cardiovascular', 'chest pain': 'cardiovascular',
            'angina': 'cardiovascular', 'AFIB': 'cardiovascular', 'cardiac failure': 'cardiovascular',
            'heart attack': 'cardiovascular', 'High blood pressure': 'cardiovascular',
            'cardiac enlargement': 'cardiovascular', 'Cardiomyopathy': 'cardiovascular',
            
            # Respiratory
            'respiratory failure': 'respiratory', 'Acute Respiratory Distress Syndrome': 'respiratory',
            'lung edema': 'respiratory', 'lung infiltration': 'respiratory', 'atelectasis': 'respiratory',
            'Difficulty breathing': 'respiratory', 'pneumonia': 'respiratory', 'neumonia': 'respiratory',
            'Apnea': 'respiratory', 'bronchitis': 'respiratory', 'asthma': 'respiratory',
            
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
            'hypokalaemia': 'metabolic', 'Blood calcium decreased': 'metabolic',
            
            # Hematological
            'Bleeding': 'hematological', 'thrombocytopenia': 'hematological', 'Leukaemia': 'hematological',
            'anaemia': 'hematological', 'deep vein thromboses': 'hematological',
            'increased white blood cell count': 'hematological',
            
            # Immunological
            'Drug hypersensitivity': 'immunological', 'allergies': 'immunological',
            'allergic dermatitis': 'immunological', 'anaphylactic reaction': 'immunological',
            
            # Hepatic
            'hepatic necrosis': 'hepatic', 'Hepatic failure': 'hepatic', 'hepatitis': 'hepatic',
            'cirrhosis': 'hepatic', 'fatty liver': 'hepatic',
            
            # Renal
            'kidney failure': 'renal', 'disorder Renal': 'renal', 'Chronic Kidney Disease': 'renal',
            'nephrocalcinosis': 'renal', 'Interstitial nephritis': 'renal'
        }
    
    def _create_drug_interaction_graph(self):
        """Create a graph representation of drug interactions"""
        print("Creating drug interaction graph...")
        
        self.drug_graph = nx.Graph()
        
        # Add edges for each drug interaction
        for _, row in self.data.iterrows():
            drug1 = row['STITCH_1']
            drug2 = row['STITCH_2']
            
            if not self.drug_graph.has_edge(drug1, drug2):
                self.drug_graph.add_edge(drug1, drug2, interactions=0, side_effects=[])
            
            self.drug_graph[drug1][drug2]['interactions'] += 1
            self.drug_graph[drug1][drug2]['side_effects'].append(row['Side_Effect_Name'])
        
        print(f"Graph created with {self.drug_graph.number_of_nodes()} nodes and {self.drug_graph.number_of_edges()} edges")
    
    def _add_graph_features(self):
        """Add graph-based features to the dataset"""
        print("Adding graph-based features...")
        
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(self.drug_graph)
        betweenness_centrality = nx.betweenness_centrality(self.drug_graph)
        closeness_centrality = nx.closeness_centrality(self.drug_graph)
        
        # Add features for drug 1
        self.data['drug1_degree_centrality'] = self.data['STITCH_1'].map(degree_centrality).fillna(0)
        self.data['drug1_betweenness_centrality'] = self.data['STITCH_1'].map(betweenness_centrality).fillna(0)
        self.data['drug1_closeness_centrality'] = self.data['STITCH_1'].map(closeness_centrality).fillna(0)
        
        # Add features for drug 2
        self.data['drug2_degree_centrality'] = self.data['STITCH_2'].map(degree_centrality).fillna(0)
        self.data['drug2_betweenness_centrality'] = self.data['STITCH_2'].map(betweenness_centrality).fillna(0)
        self.data['drug2_closeness_centrality'] = self.data['STITCH_2'].map(closeness_centrality).fillna(0)
        
        # Combined centrality features
        self.data['combined_degree_centrality'] = (self.data['drug1_degree_centrality'] + 
                                                  self.data['drug2_degree_centrality']) / 2
        self.data['combined_betweenness_centrality'] = (self.data['drug1_betweenness_centrality'] + 
                                                       self.data['drug2_betweenness_centrality']) / 2
    
    def prepare_model_data(self):
        """Prepare data for different modeling approaches"""
        print("Preparing data for modeling...")
        
        # Feature columns for different models
        basic_features = ['STITCH_1_encoded', 'STITCH_2_encoded']
        
        interaction_features = basic_features + [
            'drug_interaction_score', 'drug_sum', 'drug_diff', 'drug_ratio'
        ]
        
        advanced_features = interaction_features + [
            'drug1_total_interactions', 'drug1_unique_effects', 'drug1_partners',
            'drug2_total_interactions', 'drug2_unique_effects', 'drug2_partners',
            'drug1_degree_centrality', 'drug1_betweenness_centrality', 'drug1_closeness_centrality',
            'drug2_degree_centrality', 'drug2_betweenness_centrality', 'drug2_closeness_centrality',
            'combined_degree_centrality', 'combined_betweenness_centrality'
        ]
        
        # Handle missing values
        for feature_set in [basic_features, interaction_features, advanced_features]:
            for col in feature_set:
                if col in self.data.columns:
                    self.data[col] = self.data[col].fillna(0)
        
        # Create different feature sets
        self.feature_sets = {
            'basic': self.data[basic_features],
            'interaction': self.data[interaction_features],
            'advanced': self.data[advanced_features]
        }
        
        # Target variables
        self.targets = {
            'binary': (self.data['Side_Effect_encoded'] >= 0).astype(int),  # Binary: has side effect
            'multiclass_effect': self.data['Side_Effect_encoded'],  # Specific side effect
            'severity': self.data['severity_encoded'],  # Severity level
            'system': self.data['system_encoded']  # Affected body system
        }
        
        print(f"Feature sets prepared:")
        for name, features in self.feature_sets.items():
            print(f"  {name}: {features.shape}")
        
        print(f"Target variables prepared:")
        for name, target in self.targets.items():
            print(f"  {name}: {len(target.unique())} classes")
    
    def train_binary_classification_models(self):
        """Train binary classification models (has side effect vs no side effect)"""
        print("\n=== TRAINING BINARY CLASSIFICATION MODELS ===")
        
        # For binary classification, we need to create negative samples
        # Since all current data represents positive interactions, we'll sample negative pairs
        positive_data = self.data.copy()
        negative_data = self._generate_negative_samples(len(positive_data))
        
        # Combine positive and negative samples
        combined_data = pd.concat([positive_data, negative_data], ignore_index=True)
        
        # Prepare features and target
        X = combined_data[['STITCH_1_encoded', 'STITCH_2_encoded', 'drug_interaction_score', 
                          'drug_sum', 'drug_diff']]
        y = combined_data['has_side_effect']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        binary_models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'SVM': SVC(probability=True, random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
        }
        
        binary_results = {}
        
        for name, model in binary_models.items():
            print(f"Training {name}...")
            
            if name in ['Logistic Regression', 'SVM', 'Neural Network']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            f1 = f1_score(y_test, y_pred)
            
            binary_results[name] = {
                'model': model,
                'auc_score': auc_score,
                'f1_score': f1,
                'classification_report': classification_report(y_test, y_pred)
            }
            
            print(f"  AUC: {auc_score:.3f}, F1: {f1:.3f}")
        
        self.models['binary'] = binary_models
        self.results['binary'] = binary_results
        
        return binary_results
    
    def _generate_negative_samples(self, n_samples):
        """Generate negative samples (drug pairs with no known interactions)"""
        print("Generating negative samples...")
        
        # Get all unique drugs
        all_drugs = list(set(self.data['STITCH_1'].tolist() + self.data['STITCH_2'].tolist()))
        
        # Get existing drug pairs
        existing_pairs = set()
        for _, row in self.data.iterrows():
            pair1 = (row['STITCH_1'], row['STITCH_2'])
            pair2 = (row['STITCH_2'], row['STITCH_1'])
            existing_pairs.add(pair1)
            existing_pairs.add(pair2)
        
        # Generate random pairs that don't exist in the data
        negative_samples = []
        while len(negative_samples) < n_samples:
            drug1 = np.random.choice(all_drugs)
            drug2 = np.random.choice(all_drugs)
            
            if drug1 != drug2 and (drug1, drug2) not in existing_pairs:
                negative_samples.append({
                    'STITCH_1': drug1,
                    'STITCH_2': drug2,
                    'STITCH_1_encoded': self.encoders['stitch1'].transform([drug1])[0] 
                        if drug1 in self.encoders['stitch1'].classes_ else -1,
                    'STITCH_2_encoded': self.encoders['stitch2'].transform([drug2])[0] 
                        if drug2 in self.encoders['stitch2'].classes_ else -1,
                    'has_side_effect': 0
                })
        
        negative_df = pd.DataFrame(negative_samples)
        
        # Add interaction features
        negative_df['drug_interaction_score'] = (negative_df['STITCH_1_encoded'] * 
                                               negative_df['STITCH_2_encoded']) % 10000
        negative_df['drug_sum'] = negative_df['STITCH_1_encoded'] + negative_df['STITCH_2_encoded']
        negative_df['drug_diff'] = abs(negative_df['STITCH_1_encoded'] - negative_df['STITCH_2_encoded'])
        
        # Add positive label to existing data
        self.data['has_side_effect'] = 1
        
        return negative_df
    
    def train_multiclass_models(self):
        """Train multiclass models for specific side effect prediction"""
        print("\n=== TRAINING MULTICLASS MODELS ===")
        
        results = {}
        
        for feature_name, X in self.feature_sets.items():
            print(f"\nTraining models with {feature_name} features...")
            
            for target_name, y in self.targets.items():
                if target_name == 'binary':
                    continue
                    
                print(f"  Target: {target_name}")
                
                # Handle class imbalance with SMOTE
                smote = SMOTE(random_state=42)
                try:
                    X_resampled, y_resampled = smote.fit_resample(X, y)
                except:
                    X_resampled, y_resampled = X, y
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
                )
                
                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                # Train models
                models = {
                    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
                    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
                }
                
                for model_name, model in models.items():
                    try:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        
                        key = f"{feature_name}_{target_name}_{model_name}"
                        results[key] = {
                            'model': model,
                            'f1_score': f1,
                            'feature_set': feature_name,
                            'target': target_name
                        }
                        
                        print(f"    {model_name}: F1 = {f1:.3f}")
                        
                    except Exception as e:
                        print(f"    {model_name}: Error - {str(e)}")
        
        self.results['multiclass'] = results
        return results
    
    def save_models_and_results(self):
        """Save all trained models and results"""
        print("\nSaving models and results...")
        
        # Save encoders
        joblib.dump(self.encoders, 'e:/polyphormacy/encoders.pkl')
        joblib.dump(self.scaler, 'e:/polyphormacy/scaler.pkl')
        
        # Save models
        if 'binary' in self.models:
            for name, model in self.models['binary'].items():
                filename = f"e:/polyphormacy/binary_{name.replace(' ', '_').lower()}.pkl"
                joblib.dump(model, filename)
        
        if 'multiclass' in self.results:
            for key, result in self.results['multiclass'].items():
                if 'model' in result:
                    filename = f"e:/polyphormacy/multiclass_{key.replace(' ', '_').lower()}.pkl"
                    joblib.dump(result['model'], filename)
        
        # Save results summary
        results_summary = {}
        if 'binary' in self.results:
            results_summary['binary'] = {
                name: {'auc_score': result['auc_score'], 'f1_score': result['f1_score']}
                for name, result in self.results['binary'].items()
            }
        
        if 'multiclass' in self.results:
            results_summary['multiclass'] = {
                key: {'f1_score': result['f1_score'], 'feature_set': result['feature_set'], 'target': result['target']}
                for key, result in self.results['multiclass'].items()
                if 'f1_score' in result
            }
        
        with open('e:/polyphormacy/model_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Save processed data
        self.data.to_csv('e:/polyphormacy/processed_polypharmacy_data.csv', index=False)
        
        print("Models and results saved successfully!")
    
    def predict_interaction(self, drug1, drug2, model_type='binary', model_name='Random Forest'):
        """Predict interaction between two drugs"""
        try:
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
            
            # Scale features if needed
            if model_name in ['Logistic Regression', 'SVM', 'Neural Network']:
                features = self.scaler.transform(features)
            
            # Get model
            model = self.models[model_type][model_name]
            
            # Make prediction
            prediction = model.predict(features)[0]
            confidence = model.predict_proba(features)[0].max()
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'drug1': drug1,
                'drug2': drug2,
                'model_type': model_type,
                'model_name': model_name
            }
            
        except Exception as e:
            return {'error': str(e)}

def main():
    """Main training pipeline"""
    print("ADVANCED POLYPHARMACY PREDICTION SYSTEM")
    print("=" * 60)
    
    # Initialize predictor
    predictor = AdvancedPolypharmacyPredictor('e:/polyphormacy/polypharmacy_data.csv')
    
    # Load and preprocess data
    predictor.load_and_clean_data()
    predictor.comprehensive_feature_engineering()
    predictor.prepare_model_data()
    
    # Train models
    predictor.train_binary_classification_models()
    predictor.train_multiclass_models()
    
    # Save everything
    predictor.save_models_and_results()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("All models and results have been saved.")

if __name__ == "__main__":
    main()
