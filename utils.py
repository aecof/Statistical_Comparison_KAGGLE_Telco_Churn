import mlflow
from pprint import pprint
from imblearn.combine import SMOTETomek
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import mlflow


def yield_artifacts(run_id, path=None):
    """Yield all artifacts in the specified run"""
    client = mlflow.tracking.MlflowClient()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            yield from yield_artifacts(run_id, item.path)
        else:
            yield item.path


def fetch_logged_data(run_id):
    """Fetch params, metrics, tags, and artifacts in the specified run"""
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    # Exclude system tags: https://www.mlflow.org/docs/latest/tracking.html#system-tags
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = list(yield_artifacts(run_id))
    return {
        "params": data.params,
        "metrics": data.metrics,
        "tags": tags,
        "artifacts": artifacts,
    }



def train_val_test_split(df, train_ratio, val_ratio):
    """Divides a dataframe into train, validation, test, test

    Args:
        df (DataFrame): input DataFrame
        train_ratio (float): train instance proportions between 0 and 1
        val_ratio (float): validation instance proportions between 0 and 1

    Returns:
        tuple: Three dataframes train, validation, test
    """

    return np.split(df.sample(frac=1, random_state=42),
                    [int(train_ratio*len(df)), int((train_ratio+val_ratio)*len(df))])


def get_X_y_train_val_test(df, train_ratio, val_ratio, label_name):
    """Returns X_train, y_train, X_val, y_val, X_test, y_test given a dataframe and split ratios

    Args:
        df (DataFrame): Input dataframe
        train_ratio (float): Proportion of instance used for training between 0 and 1 
        val_ratio (float): Proportion of instance used in validation set
        label_name (str): Column name of the final label used for prediction

    Returns:
        tuple: X_train, y_train, X_val, y_val, X_test, y_test
    """
    train, validate, test = train_val_test_split(df, train_ratio, val_ratio)
    X_train, y_train = train.drop(columns=[label_name]), train[label_name]
    X_val, y_val = validate.drop(columns=[label_name]), validate[label_name]
    X_test, y_test = test.drop(columns=[label_name]), test[label_name]

    return X_train, y_train, X_val, y_val, X_test, y_test


def preprocess_fit_train_then_transform_all(X_train, X_val, X_test, numerical_columns, categorical_columns):
    """Uses sklearn pipeline to standard scale all numerical variables and one-hot encode categorical ones and

    Args:
        X_train (DataFrame):
        X_val (DataFrame): 
        X_test (DataFrame): 
        numerical_columns (list): columns name of numerical features.
        categorical_columns (list): columns name of categorical features.

    Returns:
        tuple: X_train_preprocessed, X_val_preprocessed, X_test_preprocessed
    """
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder())
                                              ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numerical_columns), ('categorical',
                                                                  categorical_transformer, categorical_columns)
        ])
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    pipeline.fit(X_train)

    train_preprocessed = pipeline.transform(X_train)
    val_preprocessed = pipeline.transform(X_val)
    test_preprocessed = pipeline.transform(X_test)

    return train_preprocessed, val_preprocessed, test_preprocessed


def train_and_hypertune():
    """Uses knowledge from data analysis and utility functions to load inputs, preprocess them, augment them, then train + hypertune two models (an SVC and a Random Forest).
    Logs info into mlruns (mlflow)

    Returns:
        Augmented dataframe (X and y), and best parameters and best scores for svc and rf 
    """
    
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    features_to_remove = ['customerID',
                          'gender',
                          'PhoneService',
                          'MultipleLines',
                          'StreamingTV',
                          'StreamingMovies',
                          'PaymentMethod',
                          'TotalCharges']

    df_selected_features = df.drop(columns=features_to_remove)
    X_train, y_train, X_val, y_val, X_test, y_test = get_X_y_train_val_test(
        df_selected_features, 0.8, 0.1, 'Churn')

    numerical_columns = ['MonthlyCharges', 'tenure']
    categorical_columns = [
        var for var in list(X_train.columns) if var not in numerical_columns]

    X_train_preprocessed, X_val_preprocessed, X_test_preprocessed = preprocess_fit_train_then_transform_all(
        X_train, X_val, X_test, numerical_columns, categorical_columns)

    y_train = y_train.apply(lambda x: int(x == 'Yes'))
    y_val = y_val.apply(lambda x: int(x == 'Yes'))
    y_test = y_test.apply(lambda x: int(x == 'Yes'))


    ################## Data Augmentation ######################
    # Since there is quite a class imbalance, lets use SMOTE Tomek to have the minority class to 80% of the majority class
    os = SMOTETomek()
    X_train_over, y_train_over = os.fit_resample(X_train_preprocessed, y_train)




    ###########################################################
    ############# First Experiment, SVC #######################
    ###########################################################
    
    mlflow.sklearn.autolog()
    parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
    svc = SVC()
    clf = GridSearchCV(svc, parameters, cv=10, scoring='f1')
    with mlflow.start_run() as run:
        clf.fit(X_train_over, y_train_over)

     # show data logged in the parent run
    print("========== parent run ==========")
    for key, data in fetch_logged_data(run.info.run_id).items():
        print("\n---------- logged {} ----------".format(key))
        pprint(data)
    # show data logged in the child runs
    filter_child_runs = "tags.mlflow.parentRunId = '{}'".format(
        run.info.run_id)
    runs = mlflow.search_runs(filter_string=filter_child_runs)
    param_cols = ["params.{}".format(p) for p in parameters.keys()]
    metric_cols = ["metrics.mean_test_score"]

    print("\n========== child runs ==========\n")
    pd.set_option("display.max_columns", None)  # prevent truncating columns
    print(runs[["run_id", *param_cols, *metric_cols]])

    ###########################################################
    ###########################################################
    ###########################################################

    ###########################################################
    ############# Second Experiment, Random Forest ############
    ###########################################################

    mlflow.sklearn.autolog()
    parameters = {"criterion": ("gini", "entropy"), "n_estimators": (50, 100)}
    rf = RandomForestClassifier()
    rf_clf = GridSearchCV(rf, parameters, cv=10, scoring='f1')
    with mlflow.start_run() as run_rf:
        rf_clf.fit(X_train_over, y_train_over)

     # show data logged in the parent run
    print("========== parent run ==========")
    for key, data in fetch_logged_data(run_rf.info.run_id).items():
        print("\n---------- logged {} ----------".format(key))
        pprint(data)
    # show data logged in the child runs
    filter_child_runs = "tags.mlflow.parentRunId = '{}'".format(
        run_rf.info.run_id)
    runs_rf = mlflow.search_runs(filter_string=filter_child_runs)
    param_cols = ["params.{}".format(p) for p in parameters.keys()]
    metric_cols = ["metrics.mean_test_score"]

    print("\n========== child runs ==========\n")
    pd.set_option("display.max_columns", None)  # prevent truncating columns
    print(runs_rf[["run_id", *param_cols, *metric_cols]])
    mlflow.sklearn.autolog(disable=True)

    ###########################################################
    ###########################################################
    ###########################################################

    return X_train_over, y_train_over, clf.best_params_, rf_clf.best_params_, clf.best_score_, rf_clf.best_score_
