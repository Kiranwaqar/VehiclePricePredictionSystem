#  Vehicle Price Prediction

This project aims to predict the price of a vehicle using various features like vehicle dimensions, age, fuel type, transmission type, and more. It uses a **Linear Regression** model trained on a cleaned and preprocessed dataset of cars. The pipeline includes handling missing values, outlier treatment, feature engineering, scaling, and model evaluation.

##  Dataset

The dataset contains various features of vehicles such as:
- Length, Width, Height
- Seating Capacity
- Fuel Tank Capacity
- Fuel Type
- Transmission
- Drivetrain
- Year, Kilometer driven
- Price

Missing values are handled, and outliers are capped using the IQR method. The `Kilometer` feature is log-transformed for normalization. A new feature `Vehicle Age` is derived from the manufacturing year.

##  Features & Preprocessing

- ✅ Null value imputation (mean/mode)
- ✅ Outlier capping (IQR method)
- ✅ Log transformation for skewed data
- ✅ Feature engineering (Vehicle Age)
- ✅ One-hot encoding for categorical features
- ✅ Feature scaling using StandardScaler
- ✅ Train/Validation/Test Split

##  Model

A simple **Linear Regression** model is trained using Scikit-Learn. Evaluation is done on train, validation, and test sets using:
- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)

##  How to Run

### 1. Clone the repo:

git clone https://github.com/Kiranwaqar/VehiclePricePredictionSystem.git

### 2. Install dependencies:

pip install -r requirements.txt

### 3. Run the script:

python vehicleprice.py

Note: Make sure the cars.csv file is placed inside the data/ directory.
