# 🚗 Car Price Prediction using Machine Learning

This project predicts the price of used cars based on various features using a Linear Regression model trained on a car dataset. It includes preprocessing steps, model training, evaluation, and prediction.

## 📂 Dataset

The dataset used is `car_price_prediction.csv`, which contains features like:

- `Price`: The target value to predict
- `Levy`, `Manufacturer`, `Model`, `Prod. year`, `Category`, `Leather interior`, `Fuel type`
- `Engine volume`, `Mileage`, `Cylinders`, `Gear box type`, `Drive wheels`
- `Doors`, `Wheel`, `Color`, `Airbags`

> Note: All missing values are filled with 0. Categorical columns are encoded using `LabelEncoder`.

---

## 🧠 Model
The project uses a **Linear Regression** model from Scikit-Learn.

### Steps:
1. Load and clean the dataset
2. Encode categorical features
3. Train the model using Linear Regression
4. Save the best performing model using `pickle`
5. Load the saved model and make predictions
---

## 🔧 Requirements
Install the required Python libraries using:
pandas
numpy
scikit-learn

## 📈 Sample Output
Predicted Price: 8500.0 | Actual Price: 9000.0
Predicted Price: 12000.0 | Actual Price: 11500.0
...
### 📁 Project Structure
├── car_price_prediction.csv     # Dataset
├── Car_Price_Prediction.py      # (Optional) Code to train and save the model and runs predictions
├── model.pickle                 # Saved trained model
├── requirements.txt
└── README.md

## 💡 Future Improvements
Use more advanced models (e.g., Random Forest, XGBoost)
Add feature normalization
Implement a web interface for predictions
Tune hyperparameters for better accuracy

