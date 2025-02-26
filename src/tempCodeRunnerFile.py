import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Get the working directory of the main.py file
working_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(working_dir)


# Step 1: Read the data
def read_data(file_name):
    file_path = f"{parent_dir}/data/{file_name}"
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        return df
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
        return df


# New function to read custom data
def read_custom_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        return df
    elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
        df = pd.read_excel(uploaded_file)
        return df
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")


# Step 2: Preprocess the data
def preprocess_data(df, target_column, scaler_type):
    # Drop rows where the target column has missing values
    df = df.dropna(subset=[target_column])

    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Check if there are only numerical or categorical columns
    numerical_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Impute missing values for numerical columns (mean imputation)
    if len(numerical_cols) > 0:
        num_imputer = SimpleImputer(strategy='mean')
        X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

        # Scale the numerical features based on scaler_type
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()

        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Impute missing values for categorical columns (mode imputation)
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

        # One-hot encode categorical features
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # Handle unknown categories
        X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
        X_test_encoded = encoder.transform(X_test[categorical_cols])
        X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(categorical_cols))
        X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(categorical_cols))
        X_train = pd.concat([X_train.drop(columns=categorical_cols), X_train_encoded], axis=1)
        X_test = pd.concat([X_test.drop(columns=categorical_cols), X_test_encoded], axis=1)

    # Ensure no missing values remain in the data
    if X_train.isnull().any().any() or X_test.isnull().any().any():
        # If missing values are found, impute them using the most frequent strategy
        final_imputer = SimpleImputer(strategy='most_frequent')
        X_train = pd.DataFrame(final_imputer.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(final_imputer.transform(X_test), columns=X_test.columns)

    # Ensure X_train, X_test, y_train, and y_test have the same number of samples
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        # If there's a mismatch, drop the extra rows from X_train and X_test
        X_train = X_train.iloc[:len(y_train)]
        X_test = X_test.iloc[:len(y_test)]

    return X_train, X_test, y_train, y_test


# Step 3: Train the model
def train_model(X_train, y_train, model, model_name):
    # Training the selected model
    model.fit(X_train, y_train)
    # Saving the trained model
    with open(f"{parent_dir}/trained_model/{model_name}.pkl", 'wb') as file:
        pickle.dump(model, file)
    return model


# Step 4: Evaluate the model and generate custom suggestions
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Generate custom suggestions based on model performance and dataset insights
    suggestions = []

    # Suggestion 1: Check for class imbalance
    class_distribution = pd.Series(y_test).value_counts(normalize=True)
    if class_distribution.min() < 0.2:  # If any class has less than 20% representation
        suggestions.append("1. **Class Imbalance Detected**: The dataset is imbalanced. Consider using techniques like SMOTE, class weighting, or collecting more data for the minority class.")

    # Suggestion 2: Check for low precision or recall
    if precision < 0.7 or recall < 0.7:
        suggestions.append("2. **Low Precision/Recall**: The model has low precision or recall. Try hyperparameter tuning, feature selection, or using a different algorithm.")

    # Suggestion 3: Check for overfitting
    if accuracy > 0.9 and f1 < 0.7:  # High accuracy but low F1 score
        suggestions.append("3. **Possible Overfitting**: The model may be overfitting. Try regularization techniques like L1/L2 regularization or reducing model complexity.")

    # Suggestion 4: Check for high confusion in the confusion matrix
    if conf_matrix.diagonal().sum() / conf_matrix.sum() < 0.7:  # Less than 70% correct predictions
        suggestions.append("4. **High Misclassification**: The model is misclassifying many samples. Consider feature engineering, hyperparameter tuning, or using ensemble methods.")

    # Suggestion 5: General advice for improving performance
    if len(suggestions) == 0:
        suggestions.append("5. **Model is Performing Well**: Consider fine-tuning hyperparameters or collecting more data for marginal improvements.")

    return accuracy, precision, recall, f1, suggestions, conf_matrix


# Step 5: Plot confusion matrix
def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    return plt


# Step 6: Plot feature importance (for tree-based models)
def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        feature_importance = pd.Series(model.feature_importances_, index=feature_names)
        feature_importance.nlargest(10).plot(kind='barh')
        plt.title("Top 10 Feature Importance")
        return plt
    else:
        return None