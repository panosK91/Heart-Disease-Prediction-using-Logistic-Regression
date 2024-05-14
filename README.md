## Heart Disease Prediction using Logistic Regression

This script performs data preprocessing, exploratory data analysis, and logistic regression to predict the risk of heart disease based on the Framingham Heart Study dataset. The steps involved are as follows:

The script starts by importing necessary libraries, including `pandas` for data manipulation, `numpy` for numerical operations, and various libraries from `sklearn` for machine learning tasks. 
It also imports `seaborn` and `matplotlib` for data visualization.

The dataset is loaded using `pd.read_csv()` and the `education` column is dropped as it is not needed for the analysis. The `male` column is renamed to `Sex_male` for better clarity. 
Rows with any missing values are removed using `dropna()`.

The script displays the first five rows of the cleaned dataset and its shape. It also prints the counts of each class in the `TenYearCHD` column to understand the distribution of the target variable. 
A count plot and a histogram are generated to visualize the target variable.

The feature matrix `X` is defined using relevant columns and the target vector `y` is defined using the `TenYearCHD` column. The features are normalized using `StandardScaler()`. 
The data is split into training and testing sets with a 70-30 ratio using `train_test_split()`.

A logistic regression model is trained on the training set using `LogisticRegression().fit()`. Predictions are made on the test set. The accuracy of the model is calculated and printed. 
A confusion matrix is generated and visualized using a heatmap. A classification report is printed to show precision, recall, and F1-score for each class.

This script provides a complete workflow for preprocessing, analyzing, and modeling a binary classification problem using logistic regression, along with the necessary visualizations and evaluation metrics.
