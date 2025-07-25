"""
Flask API for Polypharmacy Prediction System
===========================================

This module provides a REST API for the polypharmacy prediction system.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
import traceback

app = Flask(__name__)
CORS(app)

class PolypharmacyAPI:
    """API wrapper for the polypharmacy prediction system"""
    
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.scaler = None
        self.drug_info = {}
        self.load_models()
        self.load_drug_info()
    
    def load_models(self):
        """Load all trained models"""
        try:
            # Load encoders
            self.encoders = joblib.load('e:/polyphormacy/encoders.pkl')
            
            # Load models
            model_names = ['binary', 'severity', 'system']
            for model_name in model_names:
                try:
                    model_path = f'e:/polyphormacy/{model_name}_model.pkl'
                    if os.path.exists(model_path):
                        self.models[model_name] = joblib.load(model_path)
                        print(f"Loaded {model_name} model")
                except Exception as e:
                    print(f"Could not load {model_name} model: {e}")
            
            # Load results summary
            try:
                with open('e:/polyphormacy/model_results.json', 'r') as f:
                    self.model_results = json.load(f)
            except:
                self.model_results = {}
            
            print("Models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.models = {}
    
    def load_drug_info(self):
        """Load drug information from processed data"""
        try:
            data = pd.read_csv('e:/polyphormacy/polypharmacy_data.csv')
            
            # Create drug information dictionary
            for _, row in data.iterrows():
                drug1 = row['STITCH_1']
                drug2 = row['STITCH_2']
                
                if drug1 not in self.drug_info:
                    self.drug_info[drug1] = {
                        'id': drug1,
                        'interactions': [],
                        'side_effects': [],
                        'partners': set()
                    }
                
                if drug2 not in self.drug_info:
                    self.drug_info[drug2] = {
                        'id': drug2,
                        'interactions': [],
                        'side_effects': [],
                        'partners': set()
                    }
                
                # Add interaction info
                interaction = {
                    'partner': drug2,
                    'side_effect': row['Side_Effect_Name'],
                    'side_effect_code': row['UMLS_CUI_from_text']
                }
                
                self.drug_info[drug1]['interactions'].append(interaction)
                self.drug_info[drug1]['side_effects'].append(row['Side_Effect_Name'])
                self.drug_info[drug1]['partners'].add(drug2)
                
                # Add reverse interaction
                reverse_interaction = {
                    'partner': drug1,
                    'side_effect': row['Side_Effect_Name'],
                    'side_effect_code': row['UMLS_CUI_from_text']
                }
                
                self.drug_info[drug2]['interactions'].append(reverse_interaction)
                self.drug_info[drug2]['side_effects'].append(row['Side_Effect_Name'])
                self.drug_info[drug2]['partners'].add(drug1)
            
            # Convert sets to lists for JSON serialization
            for drug_id in self.drug_info:
                self.drug_info[drug_id]['partners'] = list(self.drug_info[drug_id]['partners'])
                self.drug_info[drug_id]['unique_side_effects'] = list(set(self.drug_info[drug_id]['side_effects']))
                self.drug_info[drug_id]['interaction_count'] = len(self.drug_info[drug_id]['interactions'])
            
            print(f"Drug information loaded for {len(self.drug_info)} drugs")
            
        except Exception as e:
            print(f"Error loading drug info: {e}")
            self.drug_info = {}
    
    def predict_interaction(self, drug1, drug2, model_name='random_forest'):
        """Predict drug interaction"""
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
            
            # Get model
            model_key = f'binary_{model_name}'
            if model_key not in self.models:
                return {'error': f'Model {model_name} not available'}
            
            model = self.models[model_key]
            
            # Scale features if needed
            if model_name in ['logistic_regression', 'svm', 'neural_network']:
                features = self.scaler.transform(features)
            
            # Make prediction
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            confidence = probabilities.max()
            
            # Get known interactions if they exist
            known_interactions = []
            if drug1 in self.drug_info:
                for interaction in self.drug_info[drug1]['interactions']:
                    if interaction['partner'] == drug2:
                        known_interactions.append(interaction)
            
            result = {
                'drug1': drug1,
                'drug2': drug2,
                'prediction': int(prediction),
                'confidence': float(confidence),
                'probability_no_interaction': float(probabilities[0]),
                'probability_interaction': float(probabilities[1]),
                'model_used': model_name,
                'known_interactions': known_interactions,
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {'error': f'Prediction error: {str(e)}'}
    
    def get_drug_info(self, drug_id):
        """Get information about a specific drug"""
        if drug_id in self.drug_info:
            return self.drug_info[drug_id]
        else:
            return {'error': f'Drug {drug_id} not found'}
    
    def get_all_drugs(self):
        """Get list of all available drugs"""
        return list(self.drug_info.keys())
    
    def get_model_performance(self):
        """Get model performance metrics"""
        return self.model_results

# Initialize API
api = PolypharmacyAPI()

@app.route('/')
def home():
    """API home page"""
    return jsonify({
        'message': 'Polypharmacy Prediction API',
        'version': '1.0',
        'endpoints': {
            '/predict': 'POST - Predict drug interaction',
            '/drugs': 'GET - List all drugs',
            '/drug/<drug_id>': 'GET - Get drug information',
            '/models': 'GET - Get model performance',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(api.models),
        'drugs_available': len(api.drug_info),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict drug interaction endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        drug1 = data.get('drug1')
        drug2 = data.get('drug2')
        
        if not drug1 or not drug2:
            return jsonify({'error': 'Both drug1 and drug2 must be provided'}), 400
        
        # Clean drug names if they start with CID
        if drug1.startswith('CID'):
            drug1 = drug1[3:].lstrip('0') if len(drug1) > 3 else drug1
        if drug2.startswith('CID'):
            drug2 = drug2[3:].lstrip('0') if len(drug2) > 3 else drug2
        
        # Check if drugs exist in our encoders
        drug_encoder = api.encoders.get('drug_a')
        if not drug_encoder:
            return jsonify({'error': 'Drug encoder not loaded'}), 500
        
        if drug1 not in drug_encoder.classes_ and f'CID{drug1.zfill(9)}' not in drug_encoder.classes_:
            return jsonify({'error': f'Drug {drug1} not found in database'}), 400
        if drug2 not in drug_encoder.classes_ and f'CID{drug2.zfill(9)}' not in drug_encoder.classes_:
            return jsonify({'error': f'Drug {drug2} not found in database'}), 400
        
        # Prepare features
        try:
            # Use original format if available
            drug1_encoded = drug1 if drug1 in drug_encoder.classes_ else f'CID{drug1.zfill(9)}'
            drug2_encoded = drug2 if drug2 in drug_encoder.classes_ else f'CID{drug2.zfill(9)}'
            
            # Create feature vector
            features = np.array([[
                drug_encoder.transform([drug1_encoded])[0],
                drug_encoder.transform([drug2_encoded])[0]
            ]])
            
        except Exception as e:
            return jsonify({'error': f'Error encoding drugs: {str(e)}'}), 500
        
        # Make predictions with all available models
        predictions = {}
        
        # Binary prediction (interaction or not)
        if 'binary' in api.models:
            binary_pred = api.models['binary'].predict(features)[0]
            binary_prob = api.models['binary'].predict_proba(features)[0]
            predictions['binary'] = {
                'prediction': bool(binary_pred),
                'probability': {
                    'no_interaction': float(binary_prob[0]),
                    'interaction': float(binary_prob[1])
                }
            }
        
        # Severity prediction
        if 'severity' in api.models:
            severity_pred = api.models['severity'].predict(features)[0]
            severity_prob = api.models['severity'].predict_proba(features)[0]
            severity_encoder = api.encoders.get('severity_category')
            if severity_encoder:
                severity_classes = severity_encoder.classes_
                predictions['severity'] = {
                    'prediction': severity_encoder.inverse_transform([severity_pred])[0],
                    'probabilities': {
                        cls: float(prob) for cls, prob in zip(severity_classes, severity_prob)
                    }
                }
        
        # Body system prediction
        if 'system' in api.models:
            system_pred = api.models['system'].predict(features)[0]
            system_prob = api.models['system'].predict_proba(features)[0]
            system_encoder = api.encoders.get('system_category')
            if system_encoder:
                system_classes = system_encoder.classes_
                predictions['system'] = {
                    'prediction': system_encoder.inverse_transform([system_pred])[0],
                    'probabilities': {
                        cls: float(prob) for cls, prob in zip(system_classes, system_prob)
                    }
                }
        
        return jsonify({
            'drug_pair': f"{drug1} + {drug2}",
            'predictions': predictions,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/drugs', methods=['GET'])
def get_drugs():
    """Get all available drugs"""
    drugs = api.get_all_drugs()
    return jsonify({
        'drugs': drugs,
        'count': len(drugs)
    })

@app.route('/drug/<drug_id>')
def get_drug(drug_id):
    """Get information about a specific drug"""
    info = api.get_drug_info(drug_id)
    if 'error' in info:
        return jsonify(info), 404
    return jsonify(info)

@app.route('/models')
def get_models():
    """Get model performance information"""
    return jsonify(api.get_model_performance())

@app.route('/search_drugs', methods=['GET'])
def search_drugs():
    """Search for drugs by partial name"""
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify({'drugs': []})
    
    all_drugs = api.get_all_drugs()
    matching_drugs = [drug for drug in all_drugs if query in drug.lower()]
    
    return jsonify({
        'drugs': matching_drugs[:20],  # Limit to 20 results
        'query': query
    })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction for multiple drug pairs"""
    try:
        data = request.get_json()
        
        if not data or 'drug_pairs' not in data:
            return jsonify({'error': 'drug_pairs array must be provided'}), 400
        
        drug_pairs = data['drug_pairs']
        model_name = data.get('model', 'random_forest')
        
        results = []
        for pair in drug_pairs:
            if 'drug1' in pair and 'drug2' in pair:
                result = api.predict_interaction(pair['drug1'], pair['drug2'], model_name)
                results.append(result)
            else:
                results.append({'error': 'Invalid drug pair format'})
        
        return jsonify({
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Polypharmacy Prediction API...")
    print(f"Models loaded: {len(api.models)}")
    print(f"Drugs available: {len(api.drug_info)}")
    app.run(debug=True, host='0.0.0.0', port=5000)
