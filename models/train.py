import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Rename remaining columns to match the expected feature names
    data.rename(columns={
        'romantic_encoded': 'romantic_relationship',
        'freetime': 'free_time',
        'health_encoded': 'health',
        'traveltime': 'travel_time',
        'Fedu_Encoded': 'fathers_education',
        'Medu_Encoded': 'mothers_education',
        'studytime': 'studytime',
        'failures': 'failures',
        'Gavg': 'grade_avg',
        'famrel': 'family_relationship',
        'goout': 'go_out'
    }, inplace=True)

    # Define the columns to keep (features for training)
    feature_columns = [
        'grade_avg', 'failures', 'romantic_relationship', 
        'family_relationship', 'studytime', 'health', 
        'fathers_education', 'mothers_education', 
        'free_time', 'travel_time', 'go_out'
    ]

    # Check if 'Grade' column is present
    if 'Grade' not in data.columns:
        raise KeyError("Column 'Grade' not found in the dataset after preprocessing")

    # Separate the features and target variable
    X = data[feature_columns]
    y = data['Grade']

    # Add polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Convert polynomial features to DataFrame
    feature_names = poly.get_feature_names_out(feature_columns)
    X_poly_df = pd.DataFrame(X_poly, columns=feature_names)

    # Print feature shape for debugging
    print(f"Number of features after polynomial transformation: {X_poly_df.shape[1]}")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_poly_df, y, test_size=0.2, random_state=42)

    # Standardize the numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, poly

def train_and_evaluate_random_forest(X_train, X_test, y_train, y_test):
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    rf_reg = RandomForestRegressor(random_state=42)
    rf_grid_search = GridSearchCV(estimator=rf_reg, param_grid=rf_param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
    rf_grid_search.fit(X_train, y_train)
    best_rf_reg = rf_grid_search.best_estimator_
    y_pred = best_rf_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return best_rf_reg, mse, mae, r2

def main():
    file_path = '/Users/mac/Desktop/Prediction_Dataset.csv'  # Update the path to the dataset
    X_train, X_test, y_train, y_test, scaler, poly = load_and_preprocess_data(file_path)

    # Train and evaluate Random Forest Regressor
    best_rf_reg, rf_reg_mse, rf_reg_mae, rf_reg_r2 = train_and_evaluate_random_forest(X_train, X_test, y_train, y_test)
    print(f"Random Forest Regressor MSE: {rf_reg_mse:.2f}, MAE: {rf_reg_mae:.2f}, R2: {rf_reg_r2:.2f}")

    # Save the models and scaler
    joblib.dump(best_rf_reg, '/Users/mac/Desktop/my_flask_app/models/rf_reg_model.pkl')
    joblib.dump(scaler, '/Users/mac/Desktop/my_flask_app/models/scaler.pkl')
    joblib.dump(poly, '/Users/mac/Desktop/my_flask_app/models/poly_features.pkl')

if __name__ == "__main__":
    main()
