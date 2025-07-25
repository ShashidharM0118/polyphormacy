# 🚀 RapidMiner Installation and Testing Guide

## 📋 System Requirements

### Minimum Requirements:
- **OS**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **RAM**: 4 GB minimum, 8 GB recommended
- **Storage**: 2 GB free space
- **Java**: Java 11 or higher (comes bundled with RapidMiner)

### Recommended Requirements:
- **RAM**: 16 GB or more
- **CPU**: Multi-core processor (4+ cores)
- **Storage**: SSD with 10+ GB free space

## 🔽 Download and Installation

### Step 1: Download RapidMiner Studio

1. **Visit the official website**: 
   - Go to: `https://altair.com/studio-download/`
   - Alternative: `https://rapidminer.com/` (redirects to Altair)

2. **Create a free account**:
   - Sign up with email
   - Verify your email address
   - Choose "RapidMiner Studio" (free version)

3. **Download the installer**:
   - **Windows**: `RapidMiner-Studio-installer.exe`
   - **macOS**: `RapidMiner-Studio.dmg`
   - **Linux**: `RapidMiner-Studio-installer.sh`

### Step 2: Install RapidMiner Studio

#### Windows Installation:
```powershell
# Run the installer as administrator
# Follow the installation wizard
# Default installation path: C:\Program Files\RapidMiner\RapidMiner Studio
```

#### macOS Installation:
```bash
# Mount the DMG file
# Drag RapidMiner Studio to Applications folder
# Right-click and "Open" to bypass security warnings
```

#### Linux Installation:
```bash
# Make the installer executable
chmod +x RapidMiner-Studio-installer.sh

# Run the installer
./RapidMiner-Studio-installer.sh

# Follow the installation prompts
```

### Step 3: Initial Setup

1. **Launch RapidMiner Studio**
2. **Enter license information** (free version available)
3. **Set up workspace directory**
4. **Complete the initial tutorial** (recommended)

## 🧪 Testing RapidMiner Integration

### Test 1: Run Our RapidMiner Integration

Let's test the integration we built:

```python
# Test the RapidMiner integration
python rapidminer_integration.py
```

### Test 2: Verify Installation Paths

#### Windows:
```powershell
# Check if RapidMiner is installed
ls "C:\Program Files\RapidMiner\RapidMiner Studio"

# Find the executable
ls "C:\Program Files\RapidMiner\RapidMiner Studio\scripts\rapidminer-studio.bat"
```

#### macOS:
```bash
# Check if RapidMiner is installed
ls "/Applications/RapidMiner Studio.app"

# Find the executable
ls "/Applications/RapidMiner Studio.app/Contents/MacOS/RapidMiner Studio"
```

#### Linux:
```bash
# Check installation directory (default)
ls "/opt/rapidminer-studio"

# Find the executable
ls "/opt/rapidminer-studio/bin/rapidminer-studio"
```

### Test 3: Manual RapidMiner Launch

#### Test basic RapidMiner functionality:

1. **Open RapidMiner Studio**
2. **Create a new process**
3. **Import sample data**:
   - Go to `File` → `Import Data`
   - Select CSV file
   - Use our generated `rapidminer_training_data.csv`

4. **Build a simple process**:
   - Drag "Read CSV" operator
   - Add "Set Role" operator
   - Add "Random Forest" operator
   - Connect the operators

5. **Run the process**:
   - Click the "Play" button
   - Check for any errors

## 🔧 Integration with Our System

### Step 4: Configure Our API

Update the RapidMiner path in our system:

#### For Windows:
```python
# In your Python code or API
rapidminer_path = r"C:\Program Files\RapidMiner\RapidMiner Studio\scripts\rapidminer-studio.bat"
```

#### For macOS:
```python
# In your Python code or API
rapidminer_path = "/Applications/RapidMiner Studio.app/Contents/MacOS/RapidMiner Studio"
```

#### For Linux:
```python
# In your Python code or API
rapidminer_path = "/opt/rapidminer-studio/bin/rapidminer-studio"
```

### Step 5: Test the Complete Workflow

#### 1. Start the Python API:
```bash
cd e:\polyphormacy
python api.py
```

#### 2. Start the Next.js Frontend:
```bash
cd polypharmacy-frontend
npm run dev
```

#### 3. Test the RapidMiner Integration:

1. **Open your browser**: `http://localhost:3000`
2. **Go to "RapidMiner Training" tab**
3. **Set RapidMiner path**:
   - Enter the correct path for your system
4. **Test the workflow**:
   - Click "1. Prepare Data"
   - Click "2. Create Process"
   - Click "3. Launch RapidMiner"

#### 4. Complete Training in RapidMiner:

1. **RapidMiner Studio should open** with our process
2. **Check the process**:
   - Verify data import
   - Check operators are connected
   - Review parameters
3. **Run the process**:
   - Click the play button
   - Wait for completion
   - Check results
4. **Export the model**:
   - Right-click on the model result
   - Select "Store in Repository"
   - Save as `polypharmacy_model`

## 🔍 Troubleshooting Common Issues

### Issue 1: Java Version Problems
```bash
# Check Java version
java -version

# RapidMiner requires Java 11+
# Download from: https://adoptium.net/
```

### Issue 2: Permission Errors (Windows)
```powershell
# Run PowerShell as Administrator
# Try installation again
```

### Issue 3: Path Not Found
```python
# Double-check the RapidMiner installation path
import os

# Windows
if os.path.exists(r"C:\Program Files\RapidMiner\RapidMiner Studio\scripts\rapidminer-studio.bat"):
    print("✅ RapidMiner found!")
else:
    print("❌ RapidMiner not found - check installation")
```

### Issue 4: Process File Not Loading
```python
# Check if our process file was generated
import os

if os.path.exists("rapidminer_data/polypharmacy_training.rmp"):
    print("✅ Process file exists")
    # Check file size
    size = os.path.getsize("rapidminer_data/polypharmacy_training.rmp")
    print(f"Process file size: {size} bytes")
else:
    print("❌ Process file missing - run data preparation first")
```

## 📊 Testing Results

### Expected Outputs:

#### 1. Data Preparation:
```
✅ Data prepared and saved to rapidminer_data/rapidminer_training_data.csv
✅ Features: ['STITCH_1_encoded', 'STITCH_2_encoded', 'drug_interaction_score', 'drug_sum', 'drug_diff']
✅ Encoders saved to rapidminer_data/rapidminer_encoders.pkl
```

#### 2. Process Creation:
```
✅ RapidMiner process file created: rapidminer_data/polypharmacy_training.rmp
✅ Process includes: Data import, feature selection, train/test split, Random Forest training
```

#### 3. RapidMiner Launch:
```
✅ RapidMiner Studio launching with process file
✅ Process loaded successfully
```

#### 4. Model Training (in RapidMiner):
```
✅ Data imported: 3410 rows, 5 features
✅ Training set: ~2728 samples
✅ Test set: ~682 samples
✅ Model trained successfully
✅ Performance metrics generated
```

## 🎯 Quick Test Commands

### Test the complete integration:

```bash
# 1. Navigate to project directory
cd e:\polyphormacy

# 2. Test RapidMiner integration
python -c "
from rapidminer_integration import RapidMinerIntegration
rm = RapidMinerIntegration()
rm.set_rapidminer_path('C:/Program Files/RapidMiner/RapidMiner Studio/scripts/rapidminer-studio.bat')  # Adjust path
print('🧪 Testing data preparation...')
rm.prepare_data_for_rapidminer()
print('🧪 Testing process creation...')
rm.create_rapidminer_process()
print('🧪 Testing status...')
status = rm.get_status()
for k, v in status.items():
    print(f'  {k}: {v}')
print('✅ Integration test complete!')
"

# 3. Start API server
python api.py &

# 4. Test API endpoints
curl http://localhost:5000/rapidminer/status
```

## 🔗 Alternative Options

If RapidMiner installation is challenging, you can:

### Option 1: Use RapidMiner Cloud
- Sign up at: `https://cloud.rapidminer.com/`
- No local installation required
- Limited free tier available

### Option 2: Continue with Python-only Models
- Our system already has working scikit-learn models
- RapidMiner integration is optional
- Can enhance later when convenient

### Option 3: Docker Container (Advanced)
```bash
# Pull RapidMiner Docker image (if available)
# docker pull rapidminer/rapidminer-studio
# docker run -p 8080:8080 rapidminer/rapidminer-studio
```

## 🎉 Success Indicators

You'll know everything is working when:

1. ✅ **RapidMiner Studio opens successfully**
2. ✅ **Our process file loads without errors**
3. ✅ **Data imports correctly (3410 rows)**
4. ✅ **Model trains and produces results**
5. ✅ **Web interface shows "Ready for Training: Yes"**
6. ✅ **All integration status checks pass**

## 📞 Getting Help

If you encounter issues:

1. **Check the logs** in the web interface
2. **Verify file paths** are correct for your system
3. **Ensure RapidMiner is properly installed**
4. **Check Java version compatibility**
5. **Try running RapidMiner manually first**

The system is designed to be flexible - you can use the Python models even if RapidMiner integration isn't working immediately!
