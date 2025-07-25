## 🔧 RapidMiner Manual Process Loading Instructions

### The Problem:
RapidMiner expects process files to be loaded from its internal repository system, but our generated process file exists as a regular file on disk. This causes the "Malformed repository location" error.

### The Solution:
Launch RapidMiner Studio first, then manually open the process file.

### Step-by-Step Instructions:

#### 1. Launch RapidMiner Studio
- Use the web interface: http://localhost:3000
- Go to "RapidMiner Training" tab
- Click "3. Launch RapidMiner"
- RapidMiner Studio will open (without any process loaded)

#### 2. Open the Process File Manually
In RapidMiner Studio:
1. **File** → **Open Process**
2. Navigate to: `E:\polyphormacy\rapidminer_data\polypharmacy_training.rmp`
3. Select the file and click **Open**

#### 3. Verify the Process
You should see:
- **Read CSV** operator (loading your training data)
- **Set Role** operator (setting the target variable)
- **Split Data** operator (80/20 train/test split)
- **Random Forest** operator (the machine learning model)
- **Apply Model** and **Performance** operators (for evaluation)
- **Write Model** operator (to save the trained model)

#### 4. Run the Process
1. Click the **Play** button (▶️) in the toolbar
2. Wait for the process to complete
3. Check the **Results** perspective for:
   - Model performance metrics
   - Trained Random Forest model

#### 5. Save the Model (Optional)
If you want to save the model:
1. Right-click on the model result
2. **Store in Repository**
3. Choose a name like `polypharmacy_model`

### File Locations:
- **Process File**: `E:\polyphormacy\rapidminer_data\polypharmacy_training.rmp`
- **Training Data**: `E:\polyphormacy\rapidminer_data\rapidminer_training_data.csv`
- **Output Directory**: `E:\polyphormacy\rapidminer_data\`

### Expected Results:
- **Data**: 3,410 drug interaction records
- **Features**: 5 (STITCH_1_encoded, STITCH_2_encoded, drug_interaction_score, drug_sum, drug_diff)
- **Target**: has_interaction (binary classification)
- **Model**: Random Forest classifier
- **Performance**: Classification accuracy, precision, recall, F1-score

### Troubleshooting:
- If the CSV file can't be found, check that data preparation was completed
- If operators show errors, ensure all connections are properly made
- If performance is poor, consider feature engineering or parameter tuning

This manual approach avoids the repository system issues while still allowing you to train and evaluate the model in RapidMiner Studio.
