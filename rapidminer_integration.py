"""
RapidMiner Integration Module for Polypharmacy Prediction
This module handles data export for RapidMiner and model import from RapidMiner
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
import subprocess
import xml.etree.ElementTree as ET

class RapidMinerIntegration:
    def __init__(self, data_path='polypharmacy_data.csv', output_dir='rapidminer_data'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.rapidminer_path = None  # Will be set by user
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def set_rapidminer_path(self, path):
        """Set the path to RapidMiner executable"""
        self.rapidminer_path = path
        print(f"RapidMiner path set to: {path}")
    
    def prepare_data_for_rapidminer(self):
        """
        Prepare and export data in RapidMiner-compatible format
        """
        print("Preparing data for RapidMiner...")
        
        # Load the original data - it's tab-separated with a comment line
        data = pd.read_csv(self.data_path, sep='\t', comment='#')
        data.columns = ['STITCH_1', 'STITCH_2', 'Side_Effect_Code', 'Side_Effect_Name']
        
        print(f"Loaded {len(data)} drug interaction records")
        
        # Create interaction matrix
        interactions = data.groupby(['STITCH_1', 'STITCH_2']).size().reset_index(name='interaction_count')
        
        # Generate features
        from sklearn.preprocessing import LabelEncoder
        
        # Encode drug identifiers
        le_drug1 = LabelEncoder()
        le_drug2 = LabelEncoder()
        
        interactions['STITCH_1_encoded'] = le_drug1.fit_transform(interactions['STITCH_1'])
        interactions['STITCH_2_encoded'] = le_drug2.fit_transform(interactions['STITCH_2'])
        
        # Create additional features
        interactions['drug_interaction_score'] = np.log1p(interactions['interaction_count'])
        interactions['drug_sum'] = interactions['STITCH_1_encoded'] + interactions['STITCH_2_encoded']
        interactions['drug_diff'] = abs(interactions['STITCH_1_encoded'] - interactions['STITCH_2_encoded'])
        
        # Create target variables
        interactions['has_interaction'] = (interactions['interaction_count'] > 0).astype(int)
        
        # Create severity categories (based on interaction count)
        interactions['severity_category'] = pd.cut(
            interactions['interaction_count'], 
            bins=[0, 1, 5, 10, float('inf')], 
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # Create system categories (simplified)
        interactions['system_category'] = (interactions['interaction_count'] % 3).astype(int)
        
        # Select features for RapidMiner
        feature_cols = ['STITCH_1_encoded', 'STITCH_2_encoded', 'drug_interaction_score', 'drug_sum', 'drug_diff']
        target_cols = ['has_interaction', 'severity_category', 'system_category']
        
        # Create final dataset
        rapidminer_data = interactions[feature_cols + target_cols].copy()
        
        # Save to CSV for RapidMiner
        output_file = os.path.join(self.output_dir, 'rapidminer_training_data.csv')
        rapidminer_data.to_csv(output_file, index=False)
        
        # Save encoders for later use
        encoders = {
            'drug1_encoder': le_drug1,
            'drug2_encoder': le_drug2
        }
        joblib.dump(encoders, os.path.join(self.output_dir, 'rapidminer_encoders.pkl'))
        
        # Save metadata
        metadata = {
            'features': feature_cols,
            'targets': {
                'binary': 'has_interaction',
                'severity': 'severity_category', 
                'system': 'system_category'
            },
            'samples': len(rapidminer_data),
            'created': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Data prepared and saved to {output_file}")
        print(f"Features: {metadata['features']}")
        print(f"Targets: {metadata['targets']}")
        
        return output_file, metadata
    
    def create_rapidminer_process(self):
        """
        Create a RapidMiner process file (.rmp) for training models
        """
        print("Creating RapidMiner process file...")
        
        # Get absolute paths
        current_dir = os.path.abspath('.')
        csv_file_path = os.path.join(current_dir, self.output_dir, 'rapidminer_training_data.csv').replace('\\', '/')
        model_file_path = os.path.join(current_dir, self.output_dir, 'polypharmacy_model.ioo').replace('\\', '/')
        
        print(f"Using CSV file path: {csv_file_path}")
        
        # Basic RapidMiner process XML structure
        process_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<process version="10.2.000">
  <context>
    <input/>
    <output/>
    <macros/>
  </context>
  <operator activated="true" class="process" compatibility="10.2.000" expanded="true" name="Process">
    <parameter key="logverbosity" value="init"/>
    <parameter key="random_seed" value="2001"/>
    <parameter key="send_mail" value="never"/>
    <parameter key="notification_email" value=""/>
    <parameter key="process_duration_for_mail" value="30"/>
    <parameter key="encoding" value="SYSTEM"/>
    <process expanded="true">
      
      <!-- Data Import -->
      <operator activated="true" class="read_csv" compatibility="10.2.000" expanded="true" height="68" name="Read CSV" width="90" x="112" y="85">
        <parameter key="csv_file" value="{csv_file_path}"/>
        <parameter key="column_separators" value=","/>
        <parameter key="trim_lines" value="false"/>
        <parameter key="use_quotes" value="true"/>
        <parameter key="quotes_character" value="&quot;"/>
        <parameter key="escape_character" value="\\"/>
        <parameter key="skip_comments" value="true"/>
        <parameter key="comment_characters" value="#"/>
        <parameter key="starting_row" value="1"/>
        <parameter key="parse_numbers" value="true"/>
        <parameter key="decimal_character" value="."/>
        <parameter key="grouped_digits" value="false"/>
        <parameter key="grouping_character" value=","/>
        <parameter key="infinity_representation" value=""/>
        <parameter key="date_format" value=""/>
        <parameter key="first_row_as_names" value="true"/>
        <list key="annotations"/>
        <parameter key="time_zone" value="SYSTEM"/>
        <parameter key="locale" value="English (United States)"/>
        <parameter key="encoding" value="SYSTEM"/>
        <parameter key="read_all_values_as_polynominal" value="false"/>
        <list key="data_set_meta_data_information"/>
      </operator>
      
      <!-- Set Role for Target -->
      <operator activated="true" class="set_role" compatibility="10.2.000" expanded="true" height="82" name="Set Role (Binary)" width="90" x="246" y="85">
        <parameter key="attribute_name" value="has_interaction"/>
        <parameter key="target_role" value="label"/>
        <list key="set_additional_roles"/>
      </operator>
      
      <!-- Feature Selection -->
      <operator activated="true" class="select_attributes" compatibility="10.2.000" expanded="true" height="82" name="Select Attributes" width="90" x="380" y="85">
        <parameter key="attribute_filter_type" value="subset"/>
        <parameter key="attribute" value=""/>
        <parameter key="attributes" value="STITCH_1_encoded|STITCH_2_encoded|drug_interaction_score|drug_sum|drug_diff|has_interaction"/>
        <parameter key="use_except_expression" value="false"/>
        <parameter key="value_type" value="attribute_value"/>
        <parameter key="use_value_type_exception" value="false"/>
        <parameter key="except_value_type" value="time"/>
        <parameter key="block_type" value="attribute_block"/>
        <parameter key="use_block_type_exception" value="false"/>
        <parameter key="except_block_type" value="value_matrix_row_start"/>
        <parameter key="invert_selection" value="false"/>
        <parameter key="include_special_attributes" value="true"/>
      </operator>
      
      <!-- Split Data -->
      <operator activated="true" class="split_data" compatibility="10.2.000" expanded="true" height="103" name="Split Data" width="90" x="514" y="85">
        <parameter key="relation" value="greater equals"/>
        <parameter key="value" value="0.8"/>
        <parameter key="parameter_string" value=""/>
        <parameter key="condition_class" value="attribute_value_filter"/>
        <parameter key="invert_filter" value="false"/>
        <list key="filters_list"/>
        <parameter key="filters_logic_and" value="true"/>
        <parameter key="filters_check_metadata" value="true"/>
      </operator>
      
      <!-- Random Forest Model -->
      <operator activated="true" class="random_forest" compatibility="10.2.000" expanded="true" height="103" name="Random Forest" width="90" x="648" y="34">
        <parameter key="number_of_trees" value="100"/>
        <parameter key="criterion" value="gain_ratio"/>
        <parameter key="maximal_depth" value="10"/>
        <parameter key="apply_pruning" value="false"/>
        <parameter key="confidence" value="0.1"/>
        <parameter key="apply_prepruning" value="false"/>
        <parameter key="minimal_gain" value="0.01"/>
        <parameter key="minimal_leaf_size" value="2"/>
        <parameter key="minimal_size_for_split" value="4"/>
        <parameter key="number_of_prepruning_alternatives" value="3"/>
        <parameter key="random_splits" value="false"/>
        <parameter key="guess_subset_ratio" value="true"/>
        <parameter key="subset_ratio" value="0.2"/>
        <parameter key="voting_strategy" value="confidence vote"/>
        <parameter key="use_local_random_seed" value="false"/>
        <parameter key="local_random_seed" value="1992"/>
        <parameter key="enable_parallel_execution" value="true"/>
      </operator>
      
      <!-- Apply Model -->
      <operator activated="true" class="apply_model" compatibility="10.2.000" expanded="true" height="82" name="Apply Model" width="90" x="782" y="136">
        <list key="application_parameters"/>
        <parameter key="create_view" value="false"/>
      </operator>
      
      <!-- Performance Evaluation -->
      <operator activated="true" class="performance_classification" compatibility="10.2.000" expanded="true" height="82" name="Performance" width="90" x="916" y="136">
        <parameter key="main_criterion" value="first"/>
        <parameter key="accuracy" value="true"/>
        <parameter key="classification_error" value="false"/>
        <parameter key="kappa" value="false"/>
        <parameter key="weighted_mean_recall" value="false"/>
        <parameter key="weighted_mean_precision" value="false"/>
        <parameter key="spearman_rho" value="false"/>
        <parameter key="kendall_tau" value="false"/>
        <parameter key="absolute_error" value="false"/>
        <parameter key="relative_error" value="false"/>
        <parameter key="relative_error_lenient" value="false"/>
        <parameter key="relative_error_strict" value="false"/>
        <parameter key="normalized_absolute_error" value="false"/>
        <parameter key="root_mean_squared_error" value="false"/>
        <parameter key="root_relative_squared_error" value="false"/>
        <parameter key="squared_error" value="false"/>
        <parameter key="correlation" value="false"/>
        <parameter key="squared_correlation" value="false"/>
        <parameter key="cross-entropy" value="false"/>
        <parameter key="margin" value="false"/>
        <parameter key="soft_margin_loss" value="false"/>
        <parameter key="logistic_loss" value="false"/>
        <parameter key="skip_undefined_labels" value="true"/>
        <parameter key="use_example_weights" value="true"/>
      </operator>
      
      <!-- Write Model -->
      <operator activated="true" class="write_model" compatibility="10.2.000" expanded="true" height="68" name="Write Model" width="90" x="782" y="238">
        <parameter key="model_file" value="{model_file_path}"/>
        <parameter key="overwrite_existing_file" value="true"/>
        <parameter key="create_view" value="false"/>
      </operator>
      
      <!-- Connections -->
      <connect from_op="Read CSV" from_port="output" to_op="Set Role (Binary)" to_port="example set input"/>
      <connect from_op="Set Role (Binary)" from_port="example set output" to_op="Select Attributes" to_port="example set input"/>
      <connect from_op="Select Attributes" from_port="example set output" to_op="Split Data" to_port="example set"/>
      <connect from_op="Split Data" from_port="partition 1" to_op="Random Forest" to_port="training set"/>
      <connect from_op="Split Data" from_port="partition 2" to_op="Apply Model" to_port="unlabelled data"/>
      <connect from_op="Random Forest" from_port="model" to_op="Apply Model" to_port="model"/>
      <connect from_op="Random Forest" from_port="model" to_op="Write Model" to_port="input"/>
      <connect from_op="Apply Model" from_port="labelled data" to_op="Performance" to_port="labelled data"/>
      <connect from_op="Performance" from_port="performance" to_port="result 1"/>
      <connect from_op="Write Model" from_port="through" to_port="result 2"/>
      
    </process>
  </operator>
</process>'''
        
        # Write the process file
        process_file = os.path.join(self.output_dir, 'polypharmacy_training.rmp')
        with open(process_file, 'w') as f:
            f.write(process_xml)
        
        print(f"RapidMiner process file created: {process_file}")
        return process_file
    
    def launch_rapidminer(self, process_file=None):
        """
        Launch RapidMiner Studio (without loading process file to avoid repository issues)
        """
        if not self.rapidminer_path:
            print("❌ RapidMiner path not set. Please set the path using set_rapidminer_path()")
            return False
        
        if not process_file:
            process_file = os.path.join(self.output_dir, 'polypharmacy_training.rmp')
        
        # Convert to absolute path
        abs_process_file = os.path.abspath(process_file)
        
        if not os.path.exists(abs_process_file):
            print(f"❌ Process file not found: {abs_process_file}")
            return False
        
        try:
            print(f"🚀 Launching RapidMiner Studio...")
            print(f"📁 Process file location: {abs_process_file}")
            print(f"📋 Next steps:")
            print(f"   1. In RapidMiner: File → Open Process")
            print(f"   2. Navigate to: {abs_process_file}")
            print(f"   3. Open the process file manually")
            
            # Launch RapidMiner without process file to avoid repository issues
            subprocess.Popen([self.rapidminer_path])
            return True
        except Exception as e:
            print(f"❌ Error launching RapidMiner: {e}")
            return False
    
    def load_rapidminer_model(self, model_path=None):
        """
        Load a model trained in RapidMiner (placeholder - would need actual RapidMiner model format)
        """
        if not model_path:
            model_path = os.path.join(self.output_dir, 'polypharmacy_model.ioo')
        
        if not os.path.exists(model_path):
            print(f"❌ RapidMiner model not found: {model_path}")
            return None
        
        print(f"📥 RapidMiner model found at: {model_path}")
        print("Note: Direct model loading from RapidMiner requires RapidMiner Server or additional libraries")
        return model_path
    
    def get_status(self):
        """
        Get the current status of RapidMiner integration
        """
        status = {
            'rapidminer_path_set': self.rapidminer_path is not None,
            'data_prepared': os.path.exists(os.path.join(self.output_dir, 'rapidminer_training_data.csv')),
            'process_file_exists': os.path.exists(os.path.join(self.output_dir, 'polypharmacy_training.rmp')),
            'model_exists': os.path.exists(os.path.join(self.output_dir, 'polypharmacy_model.ioo')),
            'output_dir': self.output_dir
        }
        return status

# Example usage
if __name__ == "__main__":
    rm_integration = RapidMinerIntegration()
    
    # Prepare data
    data_file, metadata = rm_integration.prepare_data_for_rapidminer()
    
    # Create process file
    process_file = rm_integration.create_rapidminer_process()
    
    # Show status
    print("\n📊 RapidMiner Integration Status:")
    status = rm_integration.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
