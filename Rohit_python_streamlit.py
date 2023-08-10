import streamlit as st

import pandas as pd
import numpy as np
import os
from io import StringIO
import io
from imblearn.over_sampling import SMOTENC
from collections import Counter
from imblearn.over_sampling import SMOTE
import base64



import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image,ImageFilter,ImageEnhance

def main():
    st.title("Data Analysis Web Application")
    st.subheader("Developed by Syed Rohit Zaman")
    st.markdown("""
    #### Description
    + Implemented a web application (used python's Library Streamlit) where dataset can be imported and the output is the result of some kind of analysis.
    """)


if __name__ == "__main__":
    main()

import streamlit as st


uploaded_file = st.file_uploader("Choose a CSV file: ")
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df1 = df.iloc[:, :-1]
    #st.write(dataframe)
else:
    df = pd.DataFrame({'A' : []})


st.subheader("Enter Number of Rows you want to show from the dataset")
Rows = st.text_input('Enter Row numbers:')
if st.button('Show:'):
  st.write(df.head(int(Rows)))



status = st.radio("Select Option:",("None","Show descriptive statistics analysis","Show Data Dimension","Check for missing values"))

if status == "None" :
       st.write("") 
if status == "Show descriptive statistics analysis" :
       st.write(df.describe()) 
if status == "Show Data Dimension" :
       st.write(df.shape) 
if status == "Check for missing values" :
       st.write(df.isnull().sum()) 



status = st.radio("Select Option for missing value handling:",("None","Drop NA","NA replace with Mean","NA replace with median"))

if status == "None" :
       st.write("") 
if status == "Drop NA" :
       df.dropna()
       st.write("Done") 
if status == "NA replace with Mean" :
       df.fillna(df1.mean())
       st.write("Done") 
if status == "NA replace with median" :
       df.fillna(df1.median())
       st.write("Done") 



# Sidebar for selecting columns
st.set_option('deprecation.showPyplotGlobalUse', False)
# Sidebar for selecting columns
# Sidebar for selecting columns
# Sidebar for selecting columns
selected_columns = st.sidebar.multiselect('Select Columns', df.columns)

# Plotting based on selected columns
if selected_columns:
    st.subheader('Selected Columns')
    st.write(selected_columns)

    plot_type = st.sidebar.selectbox('Select Plot Type', ['Line Plot', 'Bar Plot', 'Scatter Plot', 'Box Plot', 'Histogram', 'Pie Chart', 'SMOTE'])
    
    if plot_type == 'Line Plot':
        st.write("Line Plot") 
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df[selected_columns])
        st.pyplot()
    elif plot_type == 'Bar Plot':
        st.write("Bar Plot") 
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df[selected_columns])
        st.pyplot()
    elif plot_type == 'Scatter Plot':
        st.write("Scatter Plot") 
        x_column = st.sidebar.selectbox('Select X Column', selected_columns)
        y_column = st.sidebar.selectbox('Select Y Column', selected_columns)
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_column, y=y_column)
        st.pyplot()
    elif plot_type == 'Box Plot':
        st.write("Box Plot") 
        plt.figure(figsize=(10, 6))
        boxplot = sns.boxplot(data=df[selected_columns])
        st.pyplot()
        
        if st.button('Remove Outliers'):
            numeric_columns = [col for col in selected_columns if df[col].dtype != 'object']  # Exclude non-numeric columns
            for col in numeric_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
            st.success('Outliers removed successfully.')
            
            plt.figure(figsize=(10, 6))
            boxplot = sns.boxplot(data=df[selected_columns])
            st.pyplot()
            
    elif plot_type == 'Histogram':
        st.write("Histogram") 
        plt.figure(figsize=(10, 6))
        df[selected_columns].hist(bins=10, figsize=(10, 6))
        st.pyplot()
    elif plot_type == 'Pie Chart':
        st.write("Pie Chart")
        category_column = st.sidebar.selectbox('Select Categorical Column', df.columns)
        if df[category_column].dtype == 'object':
            category_counts = df[category_column].value_counts()
            plt.figure(figsize=(8, 8))
            plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
            plt.axis('equal')
            st.pyplot()
        else:
            st.warning(f"{category_column} is not a categorical column.")
    
    elif plot_type == 'SMOTE':
        st.write("SMOTE")
        target_column = st.sidebar.selectbox('Select Target Column for Oversampling', df.columns)
        if target_column in df.columns:
            unique_classes = df[target_column].nunique()
            X = df.drop(columns=[target_column])
            y = df[target_column]
            if unique_classes == 2:  # Binary classification
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                st.success('SMOTE performed successfully.')
            else:  # Multiclass case (SMOTE without categorical features)
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                st.success('SMOTE performed successfully.')
            
            st.write("Original Class Distribution:\n", y.value_counts())
            st.write("Resampled Class Distribution:\n", pd.Series(y_resampled).value_counts())
        else:
            st.error("Target column not found in the DataFrame.")
    
    if st.button('Download Modified DataFrame as CSV'):
        modified_df = df[selected_columns]  # You can modify this based on your requirement
        csv_file = modified_df.to_csv(index=False)
        b64 = base64.b64encode(csv_file.encode()).decode()  # Convert to base64
        href = f'<a href="data:file/csv;base64,{b64}" download="modified_df.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)
    
else:
    st.warning('Please select at least one column to plot from the sidebar.')



st.subheader('Random Forest')
    
# User input for features
selected_features = st.multiselect('Select Features for Random Forest', df.columns)

# User input for target variable
target_column = st.selectbox('Select Target Column for Random Forest', df.columns)

if selected_features and target_column:
        X = df[selected_features]
        y = df[target_column]

        # User input for training size
        train_size = st.slider('Training Size', min_value=0.1, max_value=0.9, value=0.7, step=0.1)

        # Splitting the dataset
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

        # User input for hyperparameters
        n_estimators = st.slider('Number of Estimators', min_value=10, max_value=200, value=100, step=10)
        max_depth = st.slider('Max Depth', min_value=1, max_value=20, value=10, step=1)

        # Training the Random Forest model
        from sklearn.ensemble import RandomForestClassifier
        rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf_model.fit(X_train, y_train)

        # Making predictions
        y_pred = rf_model.predict(X_test)

        # Displaying evaluation metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)

        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1 Score: {f1:.2f}")

        st.subheader('Confusion Matrix')
        st.write(conf_matrix)

else:
        st.warning('Please select features and target column.')


# User input for new data to predict
st.subheader('Predict New Data')
new_data = {}
for feature in selected_features:
    if df[feature].dtype == 'float64' or df[feature].dtype == 'int64':
        min_val = df[feature].min()
        max_val = df[feature].max()
        new_data[feature] = st.slider(f"Select {feature}", min_value=min_val, max_value=max_val)
    else:
        new_data[feature] = st.text_input(f"Enter value for {feature}")

# Predicting the target variable for new data
new_df = pd.DataFrame(new_data, index=[0])
predicted_target = rf_model.predict(new_df)
st.write(f"Predicted {target_column}: {predicted_target[0]}")






