# 🌊 Sea-Surface-Temperature (SST) Anomaly Prediction

This project predicts SST anomalies using LSTM neural networks and historical SST data. It includes data preprocessing, feature engineering, training, evaluation, and visualization.

## 📁 Structure
- `data/`: Dataset (NOAA or simulated)
- `models/`: Trained models
- `src/`: Scripts for preprocessing, modeling, and utility functions
- `main.py`: Main script for end-to-end prediction

## 📊 Dataset
We use SST monthly data from NOAA or synthetic data for demonstration. You can replace `sst_data.csv` with actual gridded SST time series.

## 🧠 Model
LSTM-based sequence model to predict next-month SST anomaly.

## 🚀 Usage
```bash
git clone https://github.com/yourusername/Sea-Surface-Temperature-SST-Anomaly-Prediction.git
cd Sea-Surface-Temperature-SST-Anomaly-Prediction
pip install -r requirements.txt
python main.py

