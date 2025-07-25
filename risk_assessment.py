"""
Drug Interaction Risk Assessment Tool
===================================

This module provides a comprehensive risk assessment tool for drug interactions
and side effect prediction with clinical decision support features.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

class DrugInteractionRiskAssessment:
    """
    Clinical decision support tool for drug interaction risk assessment
    """
    
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.risk_database = self.build_risk_database()
        self.clinical_guidelines = self.load_clinical_guidelines()
    
    def build_risk_database(self):
        """Build comprehensive risk database from polypharmacy data"""
        print("Building risk database...")
        
        risk_db = {}
        
        # Severity scoring
        severity_scores = {'mild': 1, 'moderate': 2, 'severe': 3}
        
        # System-based risk factors
        system_risk_factors = {
            'cardiovascular': 3,
            'respiratory': 3,
            'neurological': 2.5,
            'hematological': 3,
            'metabolic': 2,
            'gastrointestinal': 1.5,
            'immunological': 2,
            'hepatic': 2.5,
            'musculoskeletal': 1,
            'ophthalmic': 1.5
        }
        
        # Create risk profiles for each drug combination
        for _, row in self.data.iterrows():
            drug_pair = f"{row['STITCH_1']}_{row['STITCH_2']}"
            
            if drug_pair not in risk_db:
                risk_db[drug_pair] = {
                    'drugs': [row['STITCH_1'], row['STITCH_2']],
                    'side_effects': [],
                    'total_risk_score': 0,
                    'max_severity': 0,
                    'affected_systems': set(),
                    'interaction_count': 0
                }
            
            # Add side effect information
            side_effect_info = {
                'name': row['Side_Effect_Name'],
                'code': row['Polypharmacy_Side_Effect'],
                'severity': self.get_severity(row['Side_Effect_Name']),
                'system': self.get_affected_system(row['Side_Effect_Name'])
            }
            
            risk_db[drug_pair]['side_effects'].append(side_effect_info)
            risk_db[drug_pair]['interaction_count'] += 1
            risk_db[drug_pair]['affected_systems'].add(side_effect_info['system'])
            
            # Calculate risk score
            severity_score = severity_scores.get(side_effect_info['severity'], 1)
            system_factor = system_risk_factors.get(side_effect_info['system'], 1)
            
            risk_score = severity_score * system_factor
            risk_db[drug_pair]['total_risk_score'] += risk_score
            risk_db[drug_pair]['max_severity'] = max(
                risk_db[drug_pair]['max_severity'], 
                severity_score
            )
        
        # Convert sets to lists for JSON serialization
        for drug_pair in risk_db:
            risk_db[drug_pair]['affected_systems'] = list(risk_db[drug_pair]['affected_systems'])
        
        print(f"Built risk database with {len(risk_db)} drug combinations")
        return risk_db
    
    def get_severity(self, side_effect_name):
        """Get severity level for a side effect"""
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
        return severity_mapping.get(side_effect_name, 'moderate')
    
    def get_affected_system(self, side_effect_name):
        """Get affected body system for a side effect"""
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
        return system_mapping.get(side_effect_name, 'other')
    
    def load_clinical_guidelines(self):
        """Load clinical guidelines for drug interactions"""
        guidelines = {
            'contraindications': {
                'absolute': [
                    'asystole',
                    'cerebral infarct',
                    'rupture of spleen'
                ],
                'relative': [
                    'thrombocytopenia',
                    'Bleeding',
                    'lung edema',
                    'Apnea'
                ]
            },
            'monitoring_required': {
                'frequent': [
                    'hyperglycaemia',
                    'hypoglycaemia',
                    'bradycardia',
                    'hypoxia'
                ],
                'regular': [
                    'Drug hypersensitivity',
                    'allergies',
                    'abnormal movements'
                ]
            },
            'dose_adjustment': [
                'hypermagnesemia',
                'alkalosis',
                'Acidosis'
            ]
        }
        return guidelines
    
    def assess_risk(self, drug1, drug2, patient_factors=None):
        """Comprehensive risk assessment for drug combination"""
        print(f"\n=== RISK ASSESSMENT: {drug1} + {drug2} ===")
        
        drug_pair = f"{drug1}_{drug2}"
        reverse_pair = f"{drug2}_{drug1}"
        
        # Check both combinations
        risk_info = None
        if drug_pair in self.risk_database:
            risk_info = self.risk_database[drug_pair]
        elif reverse_pair in self.risk_database:
            risk_info = self.risk_database[reverse_pair]
        
        if not risk_info:
            return {
                'risk_level': 'UNKNOWN',
                'message': 'No interaction data available for this drug combination',
                'recommendation': 'Consult clinical pharmacist'
            }
        
        # Calculate risk level
        risk_score = risk_info['total_risk_score']
        max_severity = risk_info['max_severity']
        interaction_count = risk_info['interaction_count']
        
        # Determine risk level
        if max_severity >= 3 or risk_score >= 15:
            risk_level = 'HIGH'
        elif max_severity >= 2 or risk_score >= 8:
            risk_level = 'MODERATE'
        else:
            risk_level = 'LOW'
        
        # Check for contraindications
        contraindications = []
        for side_effect in risk_info['side_effects']:
            if side_effect['name'] in self.clinical_guidelines['contraindications']['absolute']:
                contraindications.append(f"ABSOLUTE: {side_effect['name']}")
            elif side_effect['name'] in self.clinical_guidelines['contraindications']['relative']:
                contraindications.append(f"RELATIVE: {side_effect['name']}")
        
        # Generate recommendations
        recommendations = self.generate_recommendations(risk_info, patient_factors)
        
        assessment = {
            'drug_combination': f"{drug1} + {drug2}",
            'risk_level': risk_level,
            'risk_score': risk_score,
            'max_severity': max_severity,
            'interaction_count': interaction_count,
            'affected_systems': risk_info['affected_systems'],
            'side_effects': risk_info['side_effects'],
            'contraindications': contraindications,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
        
        # Print summary
        print(f"Risk Level: {risk_level}")
        print(f"Risk Score: {risk_score:.1f}")
        print(f"Number of Known Interactions: {interaction_count}")
        print(f"Affected Systems: {', '.join(risk_info['affected_systems'])}")
        
        if contraindications:
            print(f"⚠️  CONTRAINDICATIONS: {'; '.join(contraindications)}")
        
        print(f"Recommendations:")
        for rec in recommendations:
            print(f"  • {rec}")
        
        return assessment
    
    def generate_recommendations(self, risk_info, patient_factors=None):
        """Generate clinical recommendations based on risk assessment"""
        recommendations = []
        
        # System-specific recommendations
        systems = risk_info['affected_systems']
        
        if 'cardiovascular' in systems:
            recommendations.append("Monitor cardiac function and vital signs")
            recommendations.append("Consider ECG monitoring")
        
        if 'respiratory' in systems:
            recommendations.append("Monitor respiratory status closely")
            recommendations.append("Ensure adequate oxygenation")
        
        if 'hematological' in systems:
            recommendations.append("Monitor complete blood count")
            recommendations.append("Watch for signs of bleeding")
        
        if 'metabolic' in systems:
            recommendations.append("Monitor blood glucose and electrolytes")
            recommendations.append("Adjust dosing based on metabolic status")
        
        if 'neurological' in systems:
            recommendations.append("Monitor neurological status")
            recommendations.append("Watch for changes in consciousness")
        
        # Severity-based recommendations
        max_severity = risk_info['max_severity']
        if max_severity >= 3:
            recommendations.append("Consider alternative drug combinations")
            recommendations.append("Implement intensive monitoring protocol")
        elif max_severity >= 2:
            recommendations.append("Implement enhanced monitoring")
            recommendations.append("Consider dose reduction")
        
        # Side effect specific recommendations
        for side_effect in risk_info['side_effects']:
            name = side_effect['name']
            
            if name in self.clinical_guidelines['monitoring_required']['frequent']:
                recommendations.append(f"Frequent monitoring for {name}")
            elif name in self.clinical_guidelines['monitoring_required']['regular']:
                recommendations.append(f"Regular monitoring for {name}")
            
            if name in self.clinical_guidelines['dose_adjustment']:
                recommendations.append(f"Consider dose adjustment due to {name} risk")
        
        # Patient factor considerations
        if patient_factors:
            if patient_factors.get('age', 0) > 65:
                recommendations.append("Elderly patient: Use lowest effective doses")
            
            if patient_factors.get('kidney_function') == 'impaired':
                recommendations.append("Adjust for renal impairment")
            
            if patient_factors.get('liver_function') == 'impaired':
                recommendations.append("Adjust for hepatic impairment")
        
        return list(set(recommendations))  # Remove duplicates
    
    def generate_risk_report(self, drug_combinations):
        """Generate comprehensive risk report for multiple drug combinations"""
        print("\n=== COMPREHENSIVE RISK REPORT ===")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'assessments': [],
            'summary': {
                'total_combinations': len(drug_combinations),
                'high_risk': 0,
                'moderate_risk': 0,
                'low_risk': 0,
                'unknown_risk': 0
            }
        }
        
        for drug1, drug2 in drug_combinations:
            assessment = self.assess_risk(drug1, drug2)
            report['assessments'].append(assessment)
            
            # Update summary
            risk_level = assessment['risk_level']
            if risk_level == 'HIGH':
                report['summary']['high_risk'] += 1
            elif risk_level == 'MODERATE':
                report['summary']['moderate_risk'] += 1
            elif risk_level == 'LOW':
                report['summary']['low_risk'] += 1
            else:
                report['summary']['unknown_risk'] += 1
        
        # Print summary
        print(f"\nSUMMARY:")
        print(f"Total combinations assessed: {report['summary']['total_combinations']}")
        print(f"High risk: {report['summary']['high_risk']}")
        print(f"Moderate risk: {report['summary']['moderate_risk']}")
        print(f"Low risk: {report['summary']['low_risk']}")
        print(f"Unknown risk: {report['summary']['unknown_risk']}")
        
        return report
    
    def save_risk_database(self):
        """Save risk database to JSON file"""
        with open('e:/polyphormacy/risk_database.json', 'w') as f:
            json.dump(self.risk_database, f, indent=2)
        print("Risk database saved to risk_database.json")
    
    def export_clinical_report(self, assessment, filename=None):
        """Export clinical report in a readable format"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'e:/polyphormacy/clinical_report_{timestamp}.txt'
        
        with open(filename, 'w') as f:
            f.write("DRUG INTERACTION RISK ASSESSMENT REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Drug Combination: {assessment['drug_combination']}\n")
            f.write(f"Assessment Date: {assessment['timestamp']}\n")
            f.write(f"Risk Level: {assessment['risk_level']}\n")
            f.write(f"Risk Score: {assessment['risk_score']:.1f}\n\n")
            
            f.write("AFFECTED SYSTEMS:\n")
            for system in assessment['affected_systems']:
                f.write(f"  • {system.title()}\n")
            f.write("\n")
            
            f.write("POTENTIAL SIDE EFFECTS:\n")
            for side_effect in assessment['side_effects']:
                f.write(f"  • {side_effect['name']} ({side_effect['severity']} severity)\n")
            f.write("\n")
            
            if assessment['contraindications']:
                f.write("CONTRAINDICATIONS:\n")
                for contraindication in assessment['contraindications']:
                    f.write(f"  ⚠️  {contraindication}\n")
                f.write("\n")
            
            f.write("CLINICAL RECOMMENDATIONS:\n")
            for rec in assessment['recommendations']:
                f.write(f"  • {rec}\n")
            f.write("\n")
            
            f.write("DISCLAIMER:\n")
            f.write("This assessment is based on available interaction data and should not\n")
            f.write("replace clinical judgment. Always consult with a clinical pharmacist\n")
            f.write("or physician for complex cases.\n")
        
        print(f"Clinical report exported to {filename}")

def main():
    """Main function for risk assessment tool"""
    print("DRUG INTERACTION RISK ASSESSMENT TOOL")
    print("=" * 50)
    
    # Initialize risk assessment tool
    risk_tool = DrugInteractionRiskAssessment('e:/polyphormacy/polypharmacy_data.csv')
    
    # Save risk database
    risk_tool.save_risk_database()
    
    # Example assessments
    print("\n=== EXAMPLE RISK ASSESSMENTS ===")
    
    # Single assessment
    assessment = risk_tool.assess_risk('CID000002173', 'CID000003345')
    
    # Export clinical report
    risk_tool.export_clinical_report(assessment)
    
    # Multiple assessments
    drug_combinations = [
        ('CID000002173', 'CID000003345'),
        # Add more combinations as they become available in your data
    ]
    
    report = risk_tool.generate_risk_report(drug_combinations)
    
    print("\n" + "=" * 50)
    print("RISK ASSESSMENT COMPLETE!")

if __name__ == "__main__":
    main()
