# Bitcoin Price Prediction using LSTM & Machine Learning

## 📌 Project Overview
The Bitcoin Price Prediction Project aims to forecast future Bitcoin closing prices using Deep Learning (LSTM Networks) and Machine Learning models. Bitcoin is highly volatile, making accurate forecasting challenging. This project applies time-series modeling, technical indicators, and neural networks to understand trends and predict future values.

## 🧠 Key Features
- Data engineering with technical indicators (MA7, MA21, STD21, returns)
- LSTM-based deep learning forecasting
- Random Forest regression for feature comparison
- Visualization of predictions and model performance
- Modular Python code structure
- Jupyter notebook walkthrough

## 📁 Repository Structure
```
bitcoin-price-prediction/
│
├── assets/                      
├── notebooks/
│   └── bitcoin_prediction.ipynb
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── models.py
│   └── train.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🧪 Technologies Used
- Python 3.10
- TensorFlow, Keras
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Docker
- Jupyter Notebook

## ⚙️ How It Works
1. Load and preprocess dataset  
2. Generate features and sequences  
3. Train LSTM model  
4. Evaluate and visualize predictions  




## 🐳 Docker Support
```
docker build -t bitcoin-prediction .
docker run -p 8888:8888 bitcoin-prediction
```

## 👤 Author
Ache Sai Ganesh
Hyderabad, India
