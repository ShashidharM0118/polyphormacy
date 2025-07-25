"""
Polypharmacy Side Effects Analysis and Prediction Models
========================================================

This script provides comprehensive analysis and modeling for polypharmacy data
to predict drug interactions and associated side effects.

Author: AI Assistant
Date: July 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

class PolypharmacyAnalyzer:
    """
    A comprehensive analyzer for polypharmacy side effects data
    """
    
    def __init__(self, data_path):
        """Initialize the analyzer with data"""
        self.data = pd.read_csv(data_path)
        self.le_stitch1 = LabelEncoder()
        self.le_stitch2 = LabelEncoder()
        self.le_side_effect = LabelEncoder()
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_and_explore_data(self):
        """Load and perform initial exploration of the data"""
        print("=== POLYPHARMACY DATA EXPLORATION ===")
        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        print("\nFirst few rows:")
        print(self.data.head())
        
        print("\nData Info:")
        print(self.data.info())
        
        print("\nUnique values:")
        for col in self.data.columns:
            print(f"{col}: {self.data[col].nunique()} unique values")
            
        print("\nMissing values:")
        print(self.data.isnull().sum())
        
        return self.data
    
    def create_features(self):
        """Create features for machine learning models"""
        print("\n=== FEATURE ENGINEERING ===")
        
        # Encode categorical variables
        self.data['STITCH_1_encoded'] = self.le_stitch1.fit_transform(self.data['STITCH_1'])
        self.data['STITCH_2_encoded'] = self.le_stitch2.fit_transform(self.data['STITCH_2'])
        self.data['Side_Effect_encoded'] = self.le_side_effect.fit_transform(self.data['Polypharmacy_Side_Effect'])
        
        # Create drug pair features
        self.data['drug_pair'] = self.data['STITCH_1'] + '_' + self.data['STITCH_2']
        self.data['drug_pair_hash'] = self.data['drug_pair'].apply(hash)
        
        # Create side effect severity categories (based on medical severity)
        severity_mapping = {
            'hypermagnesemia': 'moderate',
            'retinopathy of prematurity': 'severe',
            'atelectasis': 'severe',
            'alkalosis': 'moderate',
            'Back Ache': 'mild',
            'lung edema': 'severe',
            'agitated': 'mild',
            'abnormal movements': 'moderate',
            'Acidosis': 'severe',
            'peliosis': 'moderate',
            'rupture of spleen': 'severe',
            'Apnea': 'severe',
            'Drug hypersensitivity': 'moderate',
            'flatulence': 'mild',
            'pain in throat': 'mild',
            'allergies': 'moderate',
            'thrombocytopenia': 'severe',
            'bradycardia': 'moderate',
            'lung infiltration': 'severe',
            'Bleeding': 'severe',
            'hypoglycaemia neonatal': 'severe',
            'Gastrointestinal Obstruction': 'severe',
            'hyperglycaemia': 'moderate',
            'peritonitis': 'severe',
            'hypoglycaemia': 'moderate',
            'abdominal distension': 'mild',
            'asystole': 'severe',
            'cerebral infarct': 'severe',
            'hypoxia': 'severe',
            'Difficulty breathing': 'moderate'
        }
        
        self.data['severity'] = self.data['Side_Effect_Name'].map(severity_mapping)
        
        # Create system-based categories
        system_mapping = {
            'hypermagnesemia': 'metabolic',
            'retinopathy of prematurity': 'ophthalmic',
            'atelectasis': 'respiratory',
            'alkalosis': 'metabolic',
            'Back Ache': 'musculoskeletal',
            'lung edema': 'respiratory',
            'agitated': 'neurological',
            'abnormal movements': 'neurological',
            'Acidosis': 'metabolic',
            'peliosis': 'hepatic',
            'rupture of spleen': 'hematological',
            'Apnea': 'respiratory',
            'Drug hypersensitivity': 'immunological',
            'flatulence': 'gastrointestinal',
            'pain in throat': 'respiratory',
            'allergies': 'immunological',
            'thrombocytopenia': 'hematological',
            'bradycardia': 'cardiovascular',
            'lung infiltration': 'respiratory',
            'Bleeding': 'hematological',
            'hypoglycaemia neonatal': 'metabolic',
            'Gastrointestinal Obstruction': 'gastrointestinal',
            'hyperglycaemia': 'metabolic',
            'peritonitis': 'gastrointestinal',
            'hypoglycaemia': 'metabolic',
            'abdominal distension': 'gastrointestinal',
            'asystole': 'cardiovascular',
            'cerebral infarct': 'neurological',
            'hypoxia': 'respiratory',
            'Difficulty breathing': 'respiratory'
        }
        
        self.data['affected_system'] = self.data['Side_Effect_Name'].map(system_mapping)
        
        print(f"Created features for {len(self.data)} drug interactions")
        print(f"Severity distribution:\n{self.data['severity'].value_counts()}")
        print(f"Affected system distribution:\n{self.data['affected_system'].value_counts()}")
        
        return self.data
    
    def visualize_data(self):
        """Create visualizations for the data"""
        print("\n=== DATA VISUALIZATION ===")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Polypharmacy Side Effects Analysis', fontsize=16, fontweight='bold')
        
        # 1. Side effect severity distribution
        severity_counts = self.data['severity'].value_counts()
        axes[0, 0].pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Side Effect Severity Distribution')
        
        # 2. Affected system distribution
        system_counts = self.data['affected_system'].value_counts()
        axes[0, 1].bar(system_counts.index, system_counts.values)
        axes[0, 1].set_title('Affected Body Systems')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Top 10 most common side effects
        top_effects = self.data['Side_Effect_Name'].value_counts().head(10)
        axes[0, 2].barh(range(len(top_effects)), top_effects.values)
        axes[0, 2].set_yticks(range(len(top_effects)))
        axes[0, 2].set_yticklabels(top_effects.index)
        axes[0, 2].set_title('Most Common Side Effects')
        
        # 4. Drug combination frequency
        drug_pairs = self.data['drug_pair'].value_counts().head(10)
        axes[1, 0].bar(range(len(drug_pairs)), drug_pairs.values)
        axes[1, 0].set_title('Most Common Drug Combinations')
        axes[1, 0].set_xlabel('Drug Pair Rank')
        axes[1, 0].set_ylabel('Frequency')
        
        # 5. Severity by system heatmap
        severity_system = pd.crosstab(self.data['affected_system'], self.data['severity'])
        sns.heatmap(severity_system, annot=True, fmt='d', ax=axes[1, 1], cmap='YlOrRd')
        axes[1, 1].set_title('Severity by Affected System')
        
        # 6. Side effects per drug pair
        effects_per_pair = self.data.groupby('drug_pair').size()
        axes[1, 2].hist(effects_per_pair.values, bins=20, edgecolor='black')
        axes[1, 2].set_title('Distribution of Side Effects per Drug Pair')
        axes[1, 2].set_xlabel('Number of Side Effects')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('e:/polyphormacy/polypharmacy_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved to polypharmacy_analysis.png")
    
    def build_network_analysis(self):
        """Build and analyze drug interaction network"""
        print("\n=== NETWORK ANALYSIS ===")
        
        # Create network graph
        G = nx.Graph()
        
        # Add edges (drug pairs) with side effects as edge attributes
        for _, row in self.data.iterrows():
            G.add_edge(row['STITCH_1'], row['STITCH_2'], 
                      side_effect=row['Side_Effect_Name'],
                      severity=row['severity'],
                      system=row['affected_system'])
        
        print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Calculate network metrics
        centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
        
        # Create network dataframe
        network_df = pd.DataFrame({
            'drug': list(centrality.keys()),
            'degree_centrality': list(centrality.values()),
            'betweenness_centrality': list(betweenness.values()),
            'closeness_centrality': list(closeness.values())
        })
        
        print("Top 5 most central drugs:")
        print(network_df.nlargest(5, 'degree_centrality'))
        
        return G, network_df
    
    def prepare_modeling_data(self):
        """Prepare data for machine learning models"""
        print("\n=== PREPARING DATA FOR MODELING ===")
        
        # Features for prediction
        feature_columns = ['STITCH_1_encoded', 'STITCH_2_encoded']
        X = self.data[feature_columns]
        
        # Multiple prediction targets
        y_side_effect = self.data['Side_Effect_encoded']
        y_severity = LabelEncoder().fit_transform(self.data['severity'])
        y_system = LabelEncoder().fit_transform(self.data['affected_system'])
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target distributions:")
        print(f"- Side effects: {len(np.unique(y_side_effect))} unique")
        print(f"- Severity levels: {len(np.unique(y_severity))} unique")
        print(f"- Affected systems: {len(np.unique(y_system))} unique")
        
        return X, y_side_effect, y_severity, y_system
    
    def train_predictive_models(self):
        """Train multiple predictive models"""
        print("\n=== TRAINING PREDICTIVE MODELS ===")
        
        X, y_side_effect, y_severity, y_system = self.prepare_modeling_data()
        
        # Split data
        X_train, X_test, y_se_train, y_se_test = train_test_split(
            X, y_side_effect, test_size=0.2, random_state=42, stratify=y_side_effect
        )
        
        _, _, y_sev_train, y_sev_test = train_test_split(
            X, y_severity, test_size=0.2, random_state=42, stratify=y_severity
        )
        
        _, _, y_sys_train, y_sys_test = train_test_split(
            X, y_system, test_size=0.2, random_state=42, stratify=y_system
        )
        
        # Initialize models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Train models for each prediction task
        tasks = {
            'side_effect': (y_se_train, y_se_test),
            'severity': (y_sev_train, y_sev_test),
            'system': (y_sys_train, y_sys_test)
        }
        
        results = {}
        
        for task_name, (y_train, y_test) in tasks.items():
            print(f"\n--- Training models for {task_name} prediction ---")
            results[task_name] = {}
            
            for model_name, model in models.items():
                print(f"Training {model_name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Evaluate
                accuracy = accuracy_score(y_test, y_pred)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                
                results[task_name][model_name] = {
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'model': model
                }
                
                print(f"  Accuracy: {accuracy:.3f}")
                print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        self.results = results
        return results
    
    def predict_side_effects(self, stitch_1, stitch_2):
        """Predict side effects for a drug combination"""
        print(f"\n=== PREDICTING SIDE EFFECTS FOR {stitch_1} + {stitch_2} ===")
        
        try:
            # Encode the drug combination
            stitch_1_encoded = self.le_stitch1.transform([stitch_1])[0]
            stitch_2_encoded = self.le_stitch2.transform([stitch_2])[0]
            
            X_new = np.array([[stitch_1_encoded, stitch_2_encoded]])
            
            # Get best models for each task
            best_models = {}
            for task in self.results:
                best_accuracy = 0
                best_model = None
                for model_name, metrics in self.results[task].items():
                    if metrics['accuracy'] > best_accuracy:
                        best_accuracy = metrics['accuracy']
                        best_model = metrics['model']
                best_models[task] = best_model
            
            # Make predictions
            predictions = {}
            
            # Side effect prediction
            se_pred = best_models['side_effect'].predict(X_new)[0]
            se_proba = best_models['side_effect'].predict_proba(X_new)[0]
            predictions['side_effect'] = {
                'predicted': self.le_side_effect.inverse_transform([se_pred])[0],
                'confidence': max(se_proba)
            }
            
            # Severity prediction
            sev_pred = best_models['severity'].predict(X_new)[0]
            sev_proba = best_models['severity'].predict_proba(X_new)[0]
            severity_labels = ['mild', 'moderate', 'severe']  # Assuming this order
            predictions['severity'] = {
                'predicted': severity_labels[sev_pred],
                'confidence': max(sev_proba)
            }
            
            # System prediction
            sys_pred = best_models['system'].predict(X_new)[0]
            sys_proba = best_models['system'].predict_proba(X_new)[0]
            system_labels = ['cardiovascular', 'gastrointestinal', 'hematological', 
                           'hepatic', 'immunological', 'metabolic', 'musculoskeletal', 
                           'neurological', 'ophthalmic', 'respiratory']  # Assuming this order
            predictions['system'] = {
                'predicted': system_labels[sys_pred],
                'confidence': max(sys_proba)
            }
            
            print(f"Predicted side effect: {predictions['side_effect']['predicted']} "
                  f"(confidence: {predictions['side_effect']['confidence']:.3f})")
            print(f"Predicted severity: {predictions['severity']['predicted']} "
                  f"(confidence: {predictions['severity']['confidence']:.3f})")
            print(f"Predicted affected system: {predictions['system']['predicted']} "
                  f"(confidence: {predictions['system']['confidence']:.3f})")
            
            return predictions
            
        except ValueError as e:
            print(f"Error: Unknown drug combination. {e}")
            return None
    
    def generate_insights(self):
        """Generate medical insights from the analysis"""
        print("\n=== MEDICAL INSIGHTS ===")
        
        insights = []
        
        # Most dangerous drug combinations
        severe_effects = self.data[self.data['severity'] == 'severe']
        dangerous_pairs = severe_effects['drug_pair'].value_counts().head(5)
        insights.append(f"Most dangerous drug combinations (causing severe side effects):")
        for pair, count in dangerous_pairs.items():
            insights.append(f"  - {pair}: {count} severe side effects")
        
        # Most affected systems
        system_counts = self.data['affected_system'].value_counts()
        insights.append(f"\nMost commonly affected body systems:")
        for system, count in system_counts.head(5).items():
            insights.append(f"  - {system}: {count} side effects")
        
        # Severity distribution
        severity_pct = self.data['severity'].value_counts(normalize=True) * 100
        insights.append(f"\nSide effect severity distribution:")
        for severity, pct in severity_pct.items():
            insights.append(f"  - {severity}: {pct:.1f}%")
        
        # Model performance summary
        insights.append(f"\nModel Performance Summary:")
        for task, models in self.results.items():
            best_model = max(models.items(), key=lambda x: x[1]['accuracy'])
            insights.append(f"  - {task} prediction: {best_model[0]} "
                          f"(accuracy: {best_model[1]['accuracy']:.3f})")
        
        for insight in insights:
            print(insight)
        
        return insights
    
    def save_results(self):
        """Save analysis results to files"""
        print("\n=== SAVING RESULTS ===")
        
        # Save processed data
        self.data.to_csv('e:/polyphormacy/processed_polypharmacy_data.csv', index=False)
        
        # Save model performance
        performance_df = []
        for task, models in self.results.items():
            for model_name, metrics in models.items():
                performance_df.append({
                    'task': task,
                    'model': model_name,
                    'accuracy': metrics['accuracy'],
                    'cv_mean': metrics['cv_mean'],
                    'cv_std': metrics['cv_std']
                })
        
        pd.DataFrame(performance_df).to_csv('e:/polyphormacy/model_performance.csv', index=False)
        
        # Save insights
        insights = self.generate_insights()
        with open('e:/polyphormacy/medical_insights.txt', 'w') as f:
            for insight in insights:
                f.write(insight + '\n')
        
        print("Results saved to:")
        print("  - processed_polypharmacy_data.csv")
        print("  - model_performance.csv") 
        print("  - medical_insights.txt")
        print("  - polypharmacy_analysis.png")

def main():
    """Main analysis pipeline"""
    print("POLYPHARMACY SIDE EFFECTS ANALYSIS AND PREDICTION")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = PolypharmacyAnalyzer('e:/polyphormacy/polypharmacy_data.csv')
    
    # Run complete analysis pipeline
    analyzer.load_and_explore_data()
    analyzer.create_features()
    analyzer.visualize_data()
    analyzer.build_network_analysis()
    analyzer.train_predictive_models()
    
    # Example prediction
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTION")
    analyzer.predict_side_effects('CID000002173', 'CID000003345')
    
    # Generate insights and save results
    analyzer.save_results()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("Check the generated files for detailed results.")

if __name__ == "__main__":
    main()
