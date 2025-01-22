<div align="center">

## 🎉🎉🎉 **AutoML (Using PyCaret and Streamlit)** 🎉🎉🎉
## **A Super Low-Code Machine Learning with Interactive Web App**

<div align="left">

<div align="center">
    <img src="images/AutoML_Logo.png" alt="AutoML Logo">
</div>

# 💡 Introduction

This project is an automated machine learning (AutoML) solution that uses PyCaret and Streamlit. PyCaret is a low-code machine learning library in Python that automates machine learning work-flows, whereas Streamlit is an open-source app framework for Machine Learning and Data Science teams. This project provides an interactive web app for easy use.

## What are the Automated Machine Learning Methods to offer?

### AutoML Time Series

- Time series analysis uses statistical techniques to analyze series of data points ordered in time. 
- **Why Use:** Time series analysis is crucial for forecasting and trend analysis, which are vital in fields like finance, economics, and business.
- **Perks:** It allows businesses to understand patterns in their data and make informed future predictions, leading to better strategic decision-making.

### AutoML Regression

- Regression analysis estimates the relationship between a dependent variable and one or more independent variables.
- **Why Use:** It's used when we want to predict a continuous output variable from the input variables.
- **Perks:** It helps businesses understand how the dependent variable changes with changes in the independent variable, enabling them to make strategic decisions.

### AutoML Anomaly Detection

- Anomaly detection identifies outliers that differ significantly from the majority of the data.
- **Why Use:** It's used to detect abnormal behavior or rare events such as fraud detection in credit card transactions, or intrusion detection in network traffic.
- **Perks:** It helps businesses identify potential problems early on, preventing significant losses or damages.

### AutoML Clustering

- Clustering divides data points into several groups so that data points in the same group are more similar to each other than to those in other groups.
- **Why Use:** It's used when we want to understand the structure and patterns in data when we don't have a target variable for supervision.
- **Perks:** It helps businesses understand the segmentation in their customer base, leading to more personalized marketing and better customer service.

### AutoML Classification

- Classification identifies the category an observation belongs to based on a training dataset.
- **Why Use:** It's used when we want to predict the category of an observation.
- **Perks:** It helps businesses predict outcomes and make data-driven decisions, leading to improved services and customer satisfaction.

# 🚀 Installation

## 🌐 Option 1: Install via PyPi
AutoML (PyCaret and Streamlit) is supported on 64-bit systems with:
- Python 3.9, 3.10 and 3.11
- Ubuntu 16.04 or later
- Windows 7 or later

You can install AutoML packages (PyCaret and Streamlit) with Python's pip package manager:

```python
# install pycaret full version and streamlit
pip install pycaret[full] && pip install streamlit
```
## 📄 Option 2: Build from Source
Install the development version of the library directly from the source. The API may be unstable. It is not recommended for production use.

```python
pip install git+https://github.com/pycaret/pycaret.git@master --upgrade && pip install streamlit
```

## 📦 Option 3: Docker for PyCaret
Docker creates virtual environments with containers that keep a PyCaret installation separate from the rest of the system. PyCaret docker comes pre-installed with a Jupyter notebook. It can share resources with its host machine (access directories, use the GPU, connect to the Internet, etc.). The PyCaret Docker images are always tested for the latest major releases.

```python
# default version
docker run -p 8888:8888 pycaret/slim

# full version
docker run -p 8888:8888 pycaret/full
```

## 📦 Option 4: Docker for Streamlit
Create a separate container for Streamlit, keeping it isolated from the rest of the system. 

```Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir streamlit

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run streamlit when the container launches
CMD streamlit run app_name.py
```

Replace `app_name.py` with the name of your Streamlit app script.

To build the Docker image for Streamlit, save this Dockerfile in the same directory as your Streamlit app script and run the following command:

```Dockerfile
docker build -t streamlit_image_name .
```

Replace `streamlit_image_name` with the name you want to give to your Docker image for Streamlit.

To run the Docker container for Streamlit, use the following command:

```Dockerfile
docker run -p 8501:8501 streamlit_image_name
```

This will start the Streamlit app inside the Docker container, and it will be accessible at http://localhost:8501 in your web browser.

# 💻 Current Usage

First to have the `.conda` environment in place, you need to install Anaconda or Miniconda, which are Python distributions that include the conda package manager. Here are the steps:

1. **Download Anaconda or Miniconda**: Visit the Anaconda distribution page (https://www.anaconda.com/products/distribution) or Miniconda page (https://docs.conda.io/en/latest/miniconda.html) and download the installer for your operating system.

2. **Install Anaconda or Miniconda**: Run the installer and follow the prompts. Make sure to check the box that says "Add Anaconda to my PATH environment variable" in the installation options.

3. **Verify the Installation**: Open a new terminal window (Command Prompt on Windows) and type conda --version. If the installation was successful, you should see the version of conda printed.

4. **Create a New Conda Environment**: You can create a new conda environment `.conda` for your project using the command conda create --name .conda. To activate the environment, use the command conda activate .conda.

5. **Install PyCaret**: With your conda environment activated (or in the base environment if you didn't create a new one), install PyCaret using the command `pip install pycaret[full]`

6. **Install Streamlit**: With your conda environment activated (or in the base environment if you didn't create a new one), install Streamlit using the command `pip install streamlit`.

Now you should be able to run your Streamlit Web App. To run the Streamlit Web App, you need to use the following command:

```python
.\.conda\python.exe -m streamlit run streamlit_app.py
```

---
---

# AutoML Streamlit App

This Streamlit application provides an interface for automated machine learning (AutoML) methods (time series analysis, regression, anomaly detection, clustering, and classification).

## Styling

The application is styled for better readability and user experience. The sidebar and content background have been set to a light shade, and titles are highlighted with distinctive colors for easy navigation.

## Usage

1. **Choose AutoML Method**: Select the desired AutoML method from the sidebar menu.
2. **Read Information**: Upon selecting an AutoML method, the application displays relevant information about the method's purpose and perks.
3. **Interact with the App**: After reading the information, interact with the application to explore the chosen AutoML method.

## Details on the various functions, models and plots available in each AutoML module

### AutoML Regression

| Function                            | Explanation                                           |
|-------------------------------------|-------------------------------------------------------|
| `setup()`                           | Initializes the environment and transformation pipeline. |
| `create_model('*')`                 | Creates a regression model (e.g., 'lr' for linear regression or 'dt' for decision tree). |
| `compare_models()`                  | Compares the performance of all available regression models. |
| `ensemble_models()`                 | Ensembles regression models using techniques like boosting, bagging, and stacking. |
| `tune_model()`                      | Optimizes hyperparameters of the specified regression model. |
| `blend_models()`                    | Blends regression models using a voting mechanism. |
| `stack_models()`                    | Stacks regression models using a meta-learner. |
| `plot_model(**)`                    | Generates various plots for interpreting and evaluating regression models. |
| `evaluate_model()`                  | Evaluates the performance of the trained regression model. |
| `interpret_model()`                 | Interprets the results of the regression model, providing insights into feature importance and decision-making processes. |
| `calibrate_model()`                 | Calibrates the probabilities of the trained regression model. |
| `optimize_model()`                  | Optimizes the hyperparameters of the trained regression model. |
| `predict_model()`                   | Makes predictions on new data using the trained regression model. |
| `finalize_model()`                  | Finalizes the trained regression model for deployment. |
| `deploy_model()`                    | Deploys the trained regression model for production use. |
| `deep_check()`                      | Conducts a thorough check on the internals of the trained regression model. |
| `save_model()`                      | Saves the trained regression model to disk for later use. |
| `load_model()`                      | Loads a previously saved regression model from disk. |
| `automl()`                          | Performs automated machine learning for regression. |
| `pull()`                            | Pulls the basic experiment setup information for regression models. |
| `models()`                          | Lists all available regression models. |
| `get_metrics()`                     | Gets the performance metrics of the trained regression model. |
| `add_metric()`                      | Adds a custom performance metric to the evaluation of the regression model. |
| `remove_metric()`                   | Removes a custom performance metric from the evaluation of the regression model. |
| `get_logs()`                        | Retrieves the training logs of the regression model. |
| `get_config()`                      | Retrieves the current configuration settings for regression. |
| `set_config()`                      | Sets the configuration settings for PyCaret regression models. |
| `save_experiment()`                 | Saves the current regression experiment to disk. |
| `load_experiment()`                 | Loads a previously saved regression experiment from disk. |
| `get_leaderboard()`                 | Retrieves the leaderboard of regression models based on their performance. |
| `set_current_experiment()`          | Sets the current experiment to be used for regression. |
| `get_current_experiment()`          | Retrieves information about the current regression experiment. |
| `dashboard()`                       | Generates an interactive dashboard for analyzing and visualizing regression models. |
| `convert_model()`                   | Converts the regression model to a different framework or language. |
| `eda()`                             | Conducts exploratory data analysis on the regression dataset. |
| `check_fairness()`                  | Checks the fairness of the regression model predictions. |
| `create_api()`                      | Creates an API for the trained regression model. |
| `create_docker()`                   | Creates a Docker container for the trained regression model. |
| `create_app()`                      | Creates a web application for the trained regression model. |
| `get_allowed_engines()`             | Retrieves the allowed engines for training regression models. |
| `get_engine()`                      | Retrieves the current engine being used for training regression models. |
| `check_drift()`                     | Checks for concept drift in the regression model predictions. |

#### Regression models available

- Logistic Regression: 'lr'
- K-Nearest Neighbors: 'knn'
- Naive Bayes: 'nb'
- Decision Tree: 'dt'
- Support Vector Machine: 'svm'
- Radial Basis Function SVM: 'rbfsvm'
- Gaussian Process Classifier: 'gpc'
- Multi-Layer Perceptron: 'mlp'
- Ridge Classifier: 'ridge'
- Random Forest: 'rf'
- Quadratic Discriminant Analysis: 'qda'
- AdaBoost Classifier: 'ada'
- Gradient Boosting Classifier: 'gbc'
- Linear Discriminant Analysis: 'lda'
- Extra Trees Classifier: 'et'
- XGBoost Classifier: 'xgboost'
- LightGBM Classifier: 'lightgbm'
- CatBoost Classifier: 'catboost'

#### Plots for Regression:

- **Error Plot**: Plot showing the training and validation error as a function of model complexity or other parameters.
- **Residuals Plot**: Plot showing the distribution of residuals, or errors, of the model predictions.
- **Cook's Distance Plot**: Plot showing the influence of each observation on the regression coefficients.
- **Learning Curve**: Plot showing the model's performance on the training and validation sets as a function of the training set size.
- **Validation Curve (VC)**: Plot showing the validation score of the estimator as a function of a hyperparameter value.
- **Manifold Plot**: Plot showing the low-dimensional embedding of the data using techniques like t-SNE or PCA.
- **Feature Importance Plot**: Plot showing the importance of features in predicting the target variable.
- **Feature Importance (All Features) Plot**: Plot showing the importance of all features, including transformed or engineered features.
- **Residuals Interactive Plot**: Interactive plot showing the residuals of the model predictions.
- **Parameter Importance Plot**: Plot showing the importance of hyperparameters in a model.
- **Tree Plot**: Visualization of decision trees or tree-based models.

These plots provide insights into model performance, feature importance, decision boundaries, and other aspects relevant to regression tasks.

### AutoML Classification

#### Classification functions available

| Function                            | Explanation                                           |
|-------------------------------------|-------------------------------------------------------|
| `setup()`                           | Initializes the environment and transformation pipeline. |
| `create_model('*')`                 | Creates a classification model (e.g., 'lr' for logistic regression or 'dt' for decision tree). |
| `compare_models()`                  | Compares the performance of all available classification models. |
| `ensemble_models()`                 | Ensembles classification models using techniques like boosting, bagging, and stacking. |
| `tune_model()`                      | Optimizes hyperparameters of the specified classification model. |
| `blend_models()`                    | Blends classification models using a voting mechanism. |
| `stack_models()`                    | Stacks classification models using a meta-learner. |
| `plot_model(**)`                    | Generates various plots for interpreting and evaluating classification models. |
| `evaluate_model()`                  | Evaluates the performance of the trained classification model. |
| `interpret_model()`                 | Interprets the results of the classification model, providing insights into feature importance and decision-making processes. |
| `predict_model()`                   | Makes predictions on new data using the trained classification model. |
| `save_model()`                      | Saves the trained classification model to disk for later use. |
| `load_model()`                      | Loads a previously saved classification model from disk. |
| `pull()`                            | Pulls the basic experiment setup information for classification models. |
| `models()`                          | Lists all available classification models. |
| `dashboard()`                       | Generates an interactive dashboard for analyzing and visualizing classification models. |

#### Classification models available

- Logistic Regression: `'lr'`
- K-Nearest Neighbors: `'knn'`
- Naive Bayes: `'nb'`
- Decision Tree: `'dt'`
- Support Vector Machine: `'svm'`
- Radial Basis Function SVM: `'rbfsvm'`
- Gaussian Process Classifier: `'gpc'`
- Multi-Layer Perceptron: `'mlp'`
- Ridge Classifier: `'ridge'`
- Random Forest: `'rf'`
- Quadratic Discriminant Analysis: `'qda'`
- AdaBoost Classifier: `'ada'`
- Gradient Boosting Classifier: `'gbc'`
- Linear Discriminant Analysis: `'lda'`
- Extra Trees Classifier: `'et'`
- XGBoost Classifier: `'xgboost'`
- LightGBM Classifier: `'lightgbm'`
- CatBoost Classifier: `'catboost'`

#### Plots for Classification:

- **AUC Curve**: Area Under the Receiver Operating Characteristic Curve.
- **Threshold Curve**: Threshold plot showing the relationship between true positive rate and false positive rate at different classification thresholds.
- **Precision-Recall Curve (PR)**: Graphical representation of the precision-recall trade-off for different threshold values.
- **Class Report Plot**: Visualization of the classification report, showing precision, recall, and F1-score for each class.
- **Boundary Plot**: Plot showing decision boundaries of a classifier in feature space.
- **Recursive Feature Elimination (RFE) Plot**: Plot showing the performance of a model with different numbers of selected features in RFE.
- **Learning Curve**: Plot showing the model's performance on the training and validation sets as a function of the training set size.
- **Validation Curve (VC)**: Plot showing the validation score of the estimator as a function of a hyperparameter value.
- **Manifold Plot**: Plot showing the low-dimensional embedding of the data using techniques like t-SNE or PCA.
- **Calibration Curve**: Plot showing the calibration of predicted probabilities against true probabilities.
- **Dimensionality Reduction Plot**: Plot showing the reduced-dimensional representation of the data using dimensionality reduction techniques.
- **Feature Importance Plot**: Plot showing the importance of features in predicting the target variable.
- **Feature Importance (All Features) Plot**: Plot showing the importance of all features, including transformed or engineered features.
- **Residuals Interactive Plot**: Interactive plot showing the residuals of the model predictions.
- **Parameter Importance Plot**: Plot showing the importance of hyperparameters in a model.
- **Tree Plot**: Visualization of decision trees or tree-based models.

These plots provide insights into model performance, feature importance, decision boundaries, and other aspects relevant to classification tasks.

### AutoML Time Series

#### Functions Available

| Function                 | Explanation                                                   |
|--------------------------|---------------------------------------------------------------|
| `setup()`                | Initializes the environment for time series analysis.         |
| `create_model(*)`       | Creates a time series model.                                   |
| `compare_models()`       | Compares the performance of all available time series models.  |
| `tune_model()`           | Optimizes hyperparameters of the specified time series model. |
| `blend_models()`         | Blends time series models.                                    |
| `plot_model()`           | Generates various plots for interpreting time series models.   |
| `predict_model()`        | Makes predictions using the trained time series model.         |
| `finalize_model()`       | Finalizes the trained time series model.                       |
| `deploy_model()`         | Deploys the trained time series model.                         |
| `save_model()`           | Saves the trained time series model.                           |
| `load_model()`           | Loads a previously saved time series model.                    |
| `pull()`                 | Pulls basic experiment setup information for time series models. |
| `models()`               | Lists all available time series models.                        |
| `get_metrics()`          | Retrieves metrics for evaluating time series models.           |
| `add_metric()`           | Adds custom metrics for evaluating time series models.         |
| `remove_metric()`        | Removes custom metrics from time series models.                |
| `get_logs()`             | Retrieves logs for time series models.                         |
| `get_config()`           | Retrieves the configuration settings for time series models.   |
| `set_config()`           | Sets the configuration settings for time series models.        |
| `save_experiment()`      | Saves the experiment setup for time series models.             |
| `load_experiment()`      | Loads a saved experiment setup for time series models.         |
| `set_current_experiment()` | Sets the current experiment for time series models.           |
| `get_current_experiment()` | Retrieves information about the current experiment for time series models. |
| `check_stats()`          | Checks statistics for time series models.                      |

#### Time Series Models Available

- Naive: `'naive'`
- Grand Means: `'grand_means'`
- Seasonal Naive: `'snaive'`
- Polytrend: `'polytrend'`
- ARIMA: `'arima'`
- Auto ARIMA: `'auto_arima'`
- Exponential Smoothing: `'exp_smooth'`
- ETS (Error, Trend, Seasonality): `'ets'`
- Theta: `'theta'`
- TBATS (Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend, and Seasonal components): `'tbats'`
- BATS (Box-Cox transformation, ARMA errors, Trend, and Seasonal components): `'bats'`
- Prophet: `'prophet'`

#### Plots Available

- Time Series: `'ts'`
- Cross-Validation: `'cv'`
- AutoCorrelation Function (ACF): `'acf'`
- Partial AutoCorrelation Function (PACF): `'pacf'`
- Decomposition (STL): `'decomp_stl'`
- Diagnostics: `'diagnostics'`
- Forecast: `'forecast'`
- In-Sample Forecast: `'insample'`
- Residuals: `'residuals'`
- Train-Test Split: `'train_test_split'`
- Classical Decomposition: `'decomp_classical'`

These functions and plots are useful for conducting various aspects of time series analysis, including model creation, evaluation, and interpretation.

### AutoML Clustering

#### Functions Available

| Function                 | Explanation                                                   |
|--------------------------|---------------------------------------------------------------|
| `setup()`                | Initializes the environment for clustering and anomaly detection tasks. |
| `create_model(*)`       | Creates a clustering or anomaly detection model.              |
| `assign_model()`         | Assigns clusters or labels to data points.                    |
| `plot_model(**)`         | Generates various plots for interpreting clustering and anomaly detection models. |
| `evaluate_model()`       | Evaluates the performance of clustering and anomaly detection models. |
| `predict_model()`        | Makes predictions using the trained model.                    |
| `deploy_model()`         | Deploys the trained model.                                    |
| `save_model()`           | Saves the trained model.                                      |
| `load_model()`           | Loads a previously saved model.                               |
| `pull()`                 | Pulls basic experiment setup information for clustering and anomaly detection models. |
| `models()`               | Lists all available clustering and anomaly detection models.  |
| `get_metrics()`          | Retrieves metrics for evaluating clustering and anomaly detection models. |
| `add_metric()`           | Adds custom metrics for evaluating clustering and anomaly detection models. |
| `remove_metric()`        | Removes custom metrics from clustering and anomaly detection models. |
| `get_logs()`             | Retrieves logs for clustering and anomaly detection models.   |
| `get_config()`           | Retrieves the configuration settings for clustering and anomaly detection models. |
| `set_config()`           | Sets the configuration settings for clustering and anomaly detection models. |
| `save_experiment()`      | Saves the experiment setup for clustering and anomaly detection models. |
| `load_experiment()`      | Loads a saved experiment setup for clustering and anomaly detection models. |
| `set_current_experiment()` | Sets the current experiment for clustering and anomaly detection models. |
| `get_allowed_engines()`  | Retrieves allowed engines for clustering and anomaly detection. |
| `get_engine()`           | Retrieves the current engine for clustering and anomaly detection. |
| `get_current_experiment()` | Retrieves information about the current experiment for clustering and anomaly detection. |

#### Models Available

- K-Means: `'kmeans'`
- Affinity Propagation: `'ap'`
- Mean Shift: `'meanshift'`
- Spectral Clustering: `'sc'`
- Hierarchical Clustering: `'hclust'`
- DBSCAN: `'dbscan'`
- OPTICS: `'optics'`
- Birch: `'birch'`
- K-Modes: `'kmodes'`

#### Plots Available

- Cluster Plot: `'cluster'`
- t-SNE Plot: `'tsne'`
- Elbow Plot: `'elbow'`
- Silhouette Plot: `'silhouette'`
- Distance Plot: `'distance'`
- Distribution Plot: `'distribution'`

These functions and plots are useful for conducting various aspects of clustering tasks, including model creation, evaluation, and interpretation.

### AutoML Anomaly Detection

#### Functions Available

| Function                 | Explanation                                                   |
|--------------------------|---------------------------------------------------------------|
| `setup()`                | Initializes the environment for clustering and anomaly detection tasks. |
| `create_model()`         | Creates a clustering or anomaly detection model.              |
| `compare_models()`       | Compares the performance of all available models.             |
| `tune_model()`           | Optimizes hyperparameters of the specified model.            |
| `blend_models()`         | Blends models using techniques like voting or stacking.       |
| `plot_model()`           | Generates various plots for interpreting models.              |
| `predict_model()`        | Makes predictions using the trained model.                    |
| `finalize_model()`       | Finalizes the trained model.                                  |
| `deploy_model()`         | Deploys the trained model.                                    |
| `save_model()`           | Saves the trained model.                                      |
| `load_model()`           | Loads a previously saved model.                               |
| `pull()`                 | Pulls basic experiment setup information for models.          |
| `models()`               | Lists all available models.                                   |
| `get_metrics()`          | Retrieves metrics for evaluating models.                      |
| `add_metrics()`          | Adds custom metrics for evaluating models.                   |
| `remove_metric()`        | Removes custom metrics from models.                           |
| `get_logs()`             | Retrieves logs for models.                                    |
| `get_config()`           | Retrieves the configuration settings for models.             |
| `set_config()`           | Sets the configuration settings for models.                   |
| `save_experiment()`      | Saves the experiment setup for models.                         |
| `load_experiment()`      | Loads a saved experiment setup for models.                    |
| `set_current_experiment()` | Sets the current experiment for models.                     |
| `get_current_experiment()` | Retrieves information about the current experiment for models. |
| `check_stats()`          | Checks statistics for models.                                 |
| `get_allowed_engines()`  | Retrieves allowed engines for models.                         |
| `get_engine()`           | Retrieves the current engine for models.                      |

#### Models Available

- Angle-Based Outlier Detection (ABOD): `'abod'`
- Cluster: `'cluster'`
- Histogram-Based Outlier Detection: `'histogram'`
- K-Nearest Neighbors (KNN): `'knn'`
- Local Outlier Factor (LOF): `'lof'`
- Support Vector Machine (SVM): `'svm'`
- Principal Component Analysis (PCA): `'pca'`
- Minimum Covariance Determinant (MCD): `'mcd'`
- Subspace Outlier Detection (SOD): `'sod'`
- Stochastic Outlier Selection (SOS): `'sos'`

#### Plots Available

- t-SNE Plot: `'tsne'`
- UMAP Plot: `'umap'`

These functions and plots are useful for conducting various aspects of anomaly detection tasks, including model creation, evaluation, and interpretation.

## Details on the various parameters available in each AutoML module for setting up the experiment with various preprocessing and configuration options

### AutoML Regression

`pycaret.regression.RegressionExperiment()` function along with its parameters and explanations:

```python
pycaret.regression.RegressionExperiment(
    setup,
    data=None,
    data_func=None,
    target=-1,
    index=True,
    train_size=0.7,
    test_data=None,
    ordinal_features=None,
    numeric_features=None,
    categorical_features=None,
    date_features=None,
    text_features=None,
    ignore_features=None,
    keep_features=None,
    preprocess=True,
    create_date_columns=['day', 'month', 'year'],
    imputation_type='simple',
    numeric_imputation='mean',
    categorical_imputation='mode',
    iterative_imputation_iters=5,
    numeric_iterative_imputer='lightgbm',
    categorical_iterative_imputer='lightgbm',
    text_features_method='tf-idf',
    max_encoding_ohe=25,
    encoding_method=None,
    rare_to_value=None,
    rare_value='rare',
    polynomial_features=False,
    polynomial_degree=2,
    low_variance_threshold=None,
    group_features=None,
    group_names=None,
    drop_groups=False,
    remove_multicollinearity=False,
    multicollinearity_threshold=0.9,
    bin_numeric_features=None,
    remove_outliers=False,
    outliers_method='iforest',
    outliers_threshold='0.05',
    transformation=False,
    transformation_method='yeo-johnson',
    normalize=False,
    normalize_method='zscore',
    pca=False,
    pca_method='linear',
    pca_components=None,
    feature_selection=False,
    feature_selection_method='classic',
    feature_selection_estimator='lightgbm',
    n_features_to_select=0.2,
    transform_target=False,
    transform_target_method='yeo-johnson',
    custom_pipeline=None,
    custom_pipeline_position=-1,
    data_split_shuffle=True,
    data_split_stratify=False,
    fold_strategy='kfold',
    fold=10,
    fold_shuffle=False,
    fold_groups=None,
    n_jobs=1,
    use_gpu=False,
    html=True,
    session_id=None,
    system_log=True,
    log_experiment=False,
    experiment_name=None,
    experiment_custom_tags=None,
    log_plots=False,
    log_profile=False,
    log_data=False,
    engine=None,
    verbose=True,
    memory=True,
    profile=False,
    profile_kwargs=None
)
```

Explanation of Parameters:

- **setup**: An instance of the `pycaret.regression.setup()` function.
- **data**: DataFrame, default=None. The dataset for analysis.
- **data_func**: Function, default=None. A function to load data. If provided, the 'data' parameter is ignored.
- **target**: str, default=-1. The name of the target variable.
- **index**: bool, default=True. Whether to include the index column in the dataset.
- **train_size**: float, default=0.7. Size of the training set.
- **test_data**: DataFrame, default=None. Test dataset if separate from the main dataset.
- **ordinal_features**: dict, default=None. Dictionary of ordinal features and their categories.
- **numeric_features**: list, default=None. List of numeric features.
- **categorical_features**: list, default=None. List of categorical features.
- **date_features**: list, default=None. List of date features.
- **text_features**: list, default=None. List of text features.
- **ignore_features**: list, default=None. List of features to ignore.
- **keep_features**: list, default=None. List of features to keep.
- **preprocess**: bool, default=True. Whether to preprocess the data.
- **create_date_columns**: list, default=['day', 'month', 'year']. List of date columns to create.
- **imputation_type**: str, default='simple'. Type of imputation to perform.
- **numeric_imputation**: str, default='mean'. Imputation method for numeric features.
- **categorical_imputation**: str, default='mode'. Imputation method for categorical features.
- And so on... (continues with more parameters)

This function allows setting up a regression experiment with various preprocessing and configuration options. It prepares the dataset, preprocesses it, and sets up the environment for training and evaluating regression models.

### AutoML Classification

`pycaret.classification.ClassificationExperiment()` function along with its parameters and explanations:

```python
pycaret.classification.ClassificationExperiment(
    setup,
    data=None,
    data_func=None,
    target=-1,
    index=True,
    train_size=0.7,
    test_data=None,
    ordinal_features=None,
    numeric_features=None,
    categorical_features=None,
    date_features=None,
    text_features=None,
    ignore_features=None,
    keep_features=None,
    preprocess=True,
    create_date_columns=['day', 'month', 'year'],
    imputation_type='simple',
    numeric_imputation='mean',
    categorical_imputation='mode',
    iterative_imputation_iters=5,
    numeric_iterative_imputer='lightgbm',
    categorical_iterative_imputer='lightgbm',
    text_features_method='tf-idf',
    max_encoding_ohe=25,
    encoding_method=None,
    rare_to_value=None,
    rare_value='rare',
    polynomial_features=False,
    polynomial_degree=2,
    low_variance_threshold=None,
    group_features=None,
    group_names=None,
    drop_groups=False,
    remove_multicollinearity=False,
    multicollinearity_threshold=0.9,
    bin_numeric_features=None,
    remove_outliers=False,
    outliers_method='iforest',
    outliers_threshold='0.05',
    fix_imbalance=False,
    fix_imbalance_method='SMOTE',
    transformation=False,
    transformation_method='yeo-johnson',
    normalize=False,
    normalize_method='zscore',
    pca=False,
    pca_method='linear',
    pca_components=None,
    feature_selection=False,
    feature_selection_method='classic',
    feature_selection_estimator='lightgbm',
    n_features_to_select=0.2,
    transform_target=False,
    transform_target_method='yeo-johnson',
    custom_pipeline=None,
    custom_pipeline_position=-1,
    data_split_shuffle=True,
    data_split_stratify=False,
    fold_strategy='kfold',
    fold=10,
    fold_shuffle=False,
    fold_groups=None,
    n_jobs=1,
    use_gpu=False,
    html=True,
    session_id=None,
    system_log=True,
    log_experiment=False,
    experiment_name=None,
    experiment_custom_tags=None,
    log_plots=False,
    log_profile=False,
    log_data=False,
    engine=None,
    verbose=True,
    memory=True,
    profile=False,
    profile_kwargs=None
)
```

Explanation of Parameters:

- **setup**: An instance of the `pycaret.classification.setup()` function.
- **data**: DataFrame, default=None. The dataset for analysis.
- **data_func**: Function, default=None. A function to load data. If provided, the 'data' parameter is ignored.
- **target**: str, default=-1. The name of the target variable.
- **index**: bool, default=True. Whether to include index column in the dataset.
- **train_size**: float, default=0.7. Size of the training set.
- **test_data**: DataFrame, default=None. Test dataset if separate from the main dataset.
- **ordinal_features**: dict, default=None. Dictionary of ordinal features and their categories.
- **numeric_features**: list, default=None. List of numeric features.
- **categorical_features**: list, default=None. List of categorical features.
- **date_features**: list, default=None. List of date features.
- **text_features**: list, default=None. List of text features.
- **ignore_features**: list, default=None. List of features to ignore.
- **keep_features**: list, default=None. List of features to keep.
- **preprocess**: bool, default=True. Whether to preprocess the data.
- **create_date_columns**: list, default=['day', 'month', 'year']. List of date columns to create.
- **imputation_type**: str, default='simple'. Type of imputation to perform.
- **numeric_imputation**: str, default='mean'. Imputation method for numeric features.
- **categorical_imputation**: str, default='mode'. Imputation method for categorical features.
- And so on... (continues with more parameters)

This function allows setting up a classification experiment with various preprocessing and configuration options. It prepares the dataset, preprocesses it, and sets up the environment for training and evaluating classification models.

### AutoML Time Series Forecasting

`pycaret.time_series.TSForecastingExperiment()` function along with its parameters and explanations:

```python
pycaret.time_series.TSForecastingExperiment(
    setup,
    data=None,
    data_func=None,
    target=None,
    index=None,
    ignore_features=None,
    numeric_imputation_target=None,
    numeric_imputation_exogenous=None,
    transform_target=None,
    transform_exogenous=None,
    scale_target=None,
    scale_exogenous=None,
    fe_target_rr=None,
    fe_exogenous=None,
    fold_strategy='expanding',
    fold=3,
    fh=1,
    hyperparameter_split='all',
    seasonal_period=None,
    ignore_seasonality_test=False,
    sp_detection='auto',
    max_sp_to_consider=60,
    remove_harmonics=False,
    harmonic_order_method='harmonic_max',
    num_sps_to_use=1,
    point_alpha=None,
    coverage=0.9,
    enforce_exogenous=True,
    n_jobs=-1,
    use_gpu=False,
    custom_pipeline=None,
    html=True,
    session_id=None,
    system_log=True,
    log_experiment=False,
    experiment_name=None,
    experiment_custom_tags=None,
    log_plots=False,
    log_profile=False,
    log_data=False,
    engine=None,
    verbose=True,
    profile=False,
    profile_kwargs=None,
    fig_kwargs=None
)
```

Explanation of Parameters:

- **setup**: An instance of the `pycaret.time_series.setup()` function.
- **data**: DataFrame, default=None. The dataset for analysis.
- **data_func**: Function, default=None. A function to load data. If provided, the 'data' parameter is ignored.
- **target**: str, default=None. The name of the target variable.
- **index**: str or int, default=None. The index column of the time series data.
- **ignore_features**: list, default=None. List of features to ignore during modeling.
- **numeric_imputation_target**: str, default=None. Imputation method for missing values in the target variable.
- **numeric_imputation_exogenous**: str, default=None. Imputation method for missing values in exogenous variables.
- **transform_target**: str, default=None. Transformation method for the target variable.
- **transform_exogenous**: str, default=None. Transformation method for exogenous variables.
- **scale_target**: str, default=None. Scaling method for the target variable.
- **scale_exogenous**: str, default=None. Scaling method for exogenous variables.
- **fe_target_rr**: int, default=None. Forecasting horizon for feature extraction of the target variable.
- **fe_exogenous**: int, default=None. Forecasting horizon for feature extraction of exogenous variables.
- **fold_strategy**: str, default='expanding'. The folding strategy for time series cross-validation.
- **fold**: int, default=3. The number of folds for time series cross-validation.
- And so on... (continues with more parameters)

This function allows setting up a time series forecasting experiment with various preprocessing and configuration options. It prepares the dataset, preprocesses it, and sets up the environment for training and evaluating time series forecasting models.

### AutoML Clustering

`pycaret.clustering.ClusteringExperiment()` function along with its parameters and explanations:

```python
pycaret.clustering.ClusteringExperiment(
    setup,
    data=None,
    data_func=None,
    index=True,
    ordinal_features=None,
    numeric_features=None,
    categorical_features=None,
    date_features=None,
    text_features=None,
    ignore_features=None,
    keep_features=None,
    preprocess=True,
    create_date_columns=['day', 'month', 'year'],
    imputation_type='simple',
    numeric_imputation='mean',
    categorical_imputation='mode',
    text_features_method='tf-idf',
    max_encoding_ohe=-1,
    encoding_method=None,
    rare_to_value=None,
    rare_value='rare',
    polynomial_features=False,
    polynomial_degree=2,
    low_variance_threshold=None,
    remove_multicollinearity=False,
    multicollinearity_threshold=0.9,
    bin_numeric_features=None,
    remove_outliers=False,
    outliers_method='iforest',
    outliers_threshold=0.05,
    transformation=False,
    transformation_method='yeo-johnson',
    normalize=False,
    normalize_method='zscore',
    pca=False,
    pca_method='linear',
    pca_components=None,
    custom_pipeline=None,
    custom_pipeline_position=-1,
    n_jobs=-1,
    use_gpu=False,
    html=True,
    session_id=None,
    system_log=True,
    log_experiment=False,
    experiment_name=None,
    experiment_custom_tags=None,
    log_plots=False,
    log_profile=False,
    log_data=False,
    verbose=True,
    memory=True,
    profile=False,
    profile_kwargs=None
)
```

Explanation of Parameters:

- **setup**: An instance of the `pycaret.clustering.setup()` function.
- **data**: DataFrame, default=None. The dataset for analysis.
- **data_func**: Function, default=None. A function to load data. If provided, the 'data' parameter is ignored.
- **index**: bool, default=True. Whether to include index column in the dataset.
- **ordinal_features**: dict, default=None. Dictionary of ordinal features and their categories.
- **numeric_features**: list, default=None. List of numeric features.
- **categorical_features**: list, default=None. List of categorical features.
- **date_features**: list, default=None. List of date features.
- **text_features**: list, default=None. List of text features.
- **ignore_features**: list, default=None. List of features to ignore.
- **keep_features**: list, default=None. List of features to keep.
- **preprocess**: bool, default=True. Whether to preprocess the data.
- **create_date_columns**: list, default=['day', 'month', 'year']. List of date columns to create.
- **imputation_type**: str, default='simple'. Type of imputation to perform.
- And so on... (continues with more parameters)

This function allows setting up a clustering experiment with various preprocessing and configuration options. It prepares the dataset, preprocesses it, and sets up the environment for training and evaluating clustering models.

### AutoML Anomaly Detection

`pycaret.anomaly.AnomalyExperiment()` function along with its parameters and explanations:

```python
pycaret.anomaly.AnomalyExperiment(
    setup,
    data=None,
    data_func=None,
    index=True,
    ordinal_features=None,
    numeric_features=None,
    categorical_features=None,
    date_features=None,
    text_features=None,
    ignore_features=None,
    keep_features=None,
    preprocess=True,
    create_date_columns=['day', 'month', 'year'],
    imputation_type='simple',
    numeric_imputation='mean',
    categorical_imputation='mode',
    text_features_method='tf-idf',
    max_encoding_ohe=-1,
    encoding_method=None,
    rare_to_value=None,
    rare_value='rare',
    polynomial_features=False,
    polynomial_degree=2,
    low_variance_threshold=None,
    group_features=None,
    group_names=None,
    drop_groups=False,
    remove_multicollinearity=False,
    multicollinearity_threshold=0.9,
    bin_features=None,
    remove_outliers=False,
    outliers_method='iforest',
    outliers_threshold=0.05,
    transformation=False,
    transformation_method='yeo-johnson',
    normalize=False,
    normalize_method='zscore',
    pca=False,
    pca_method='linear',
    pca_components=None,
    custom_pipeline=None,
    custom_pipeline_position=-1,
    n_jobs=-1,
    use_gpu=False,
    html=True,
    session_id=None,
    system_log=True,
    log_experiment=False,
    experiment_name=None,
    experiment_custom_tags=None,
    log_plots=False,
    log_profile=False,
    log_data=False,
    verbose=True,
    memory=True,
    profile=False,
    profile_kwargs=None
)
```

Explanation of Parameters:

- **setup**: An instance of the `pycaret.anomaly.setup()` function.
- **data**: DataFrame, default=None. The dataset for analysis.
- **data_func**: Function, default=None. A function to load data. If provided, the 'data' parameter is ignored.
- **index**: bool, default=True. Whether to include index column in the dataset.
- **ordinal_features**: dict, default=None. Dictionary of ordinal features and their categories.
- **numeric_features**: list, default=None. List of numeric features.
- **categorical_features**: list, default=None. List of categorical features.
- **date_features**: list, default=None. List of date features.
- **text_features**: list, default=None. List of text features.
- **ignore_features**: list, default=None. List of features to ignore.
- **keep_features**: list, default=None. List of features to keep.
- **preprocess**: bool, default=True. Whether to preprocess the data.
- **create_date_columns**: list, default=['day', 'month', 'year']. List of date columns to create.
- **imputation_type**: str, default='simple'. Type of imputation to perform.
- And so on... (continues with more parameters)

This function allows setting up an anomaly detection experiment with various preprocessing and configuration options. It prepares the dataset, preprocesses it, and sets up the environment for training and evaluating anomaly detection models.