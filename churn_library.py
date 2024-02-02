# library doc string
'''
Author: Benjamin Knöpfle
E-Mail: benjamin.knoepfle@t-systems.com
Date: 29.01.2024

This module contains all functions needed to find customers who are likely to churn
'''

#############################################################
#                                                           #
#                    import libraries                       #
#                                                           #
#############################################################
import os
import logging
from collections.abc import Iterable

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shap

import constants as cnst

####### setup globals and parameters for libraries###########
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename=cnst.LOGFILE,
    level=logging.DEBUG,
    filemode='w',
    format='%(asctime)s, %(name)s - %(levelname)s - %(message)s [%(funcName)s]',
    datefmt='%d.%m.%Y %I:%M:%S')


sns.set()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


##############################################################
#                                                            #
#                 functionality implementation               #
#                                                            #
##############################################################


def import_data(path_to_data):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        logging.info("Import data from %s", path_to_data)
        imported_data = pd.read_csv(path_to_data)
        logging.debug(
            "DataFrame has %i rows and %i columns",
            imported_data.shape[0],
            imported_data.shape[1])
        if imported_data.shape[0] == 0 | imported_data.shape[1] == 0:
            logging.error(
                "ERROR: Testing import_data: The file doesn't appear to have rows or columns")
            raise pd.errors.EmptyDataError

        logging.info("SUCCESS")
        return imported_data
    except FileNotFoundError as err:
        logging.error("ERROR: File %s not found!", path_to_data)
        logging.error(err)
        raise err

################ exploratory data analysis (EDA) ##########


def perform_eda(data_frame, file_path=cnst.EDA_IMAGES_PATH):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe
            file_path: string path to store analysis report to
    output:
            None
    '''
    try:
        if not isinstance(data_frame, pd.DataFrame):
            logging.error('ERROR: data_frame is not of type pd.DataFrame')
            logging.error('ERROR: data_frame is of type %s', type(data_frame))
            raise TypeError
        if not os.path.isdir(file_path):
            logging.error('ERROR: file_path %s does not exist', file_path)
            raise FileNotFoundError

        perform_initial_analysis(data_frame, file_path)
        perform_univariate_analysis(data_frame, file_path)
        perform_correlation_analysis(data_frame, file_path)
    except KeyError as err:
        logging.error('ERROR: Expected column is missing in dataframe')
        logging.error(err)
        raise err


def perform_initial_analysis(data_frame, file_path=cnst.EDA_IMAGES_PATH):
    '''
    perform initial anlaysis on data_frame and logs it
    input:
            data_frame: pandas dataframe
            file_path: string path to store analysis report to
    output:
            None
    '''
    logging.info("Performing initial Analysis")

    rows, cols = data_frame.shape

    report = [f'DataFrame has {rows} rows and {cols} columns',
              "Null Values within columns:",
              str(data_frame.isnull().sum()),
              "Simple statistics:",
              str(data_frame[cnst.QUANT_FEATURES].describe().T),
              "First few lines of the DataFrame:",
              str(data_frame.head().T)]
    report = '\n'.join(report)

    with open(f'{file_path}initial_analysis.txt', 'w') as file:
        file.write(report)
    logging.info("SUCCESS")


def perform_univariate_analysis(data_frame,
                                file_path=cnst.EDA_IMAGES_PATH):
    '''
    perform univariate anlaysis on data_frame by creating a plot
    input:
            data_frame: pandas dataframe
            analysis_list: list of dictionarys assigning analysis to column [{col: ,kind:}....]
                col: string name of the column
                kind: string kind of the analysis [hist, counts, density]
            file_path: string path to store analysis figures to
    output:
            None
    '''
    logging.info("Performing univariate analysis")
    for action in cnst.ANALYST_LIST:
        logging.debug(
            "DEBUG: Column is %s with kind of analysis %s",
            action['col'],
            action['kind'])
        if action['kind'] == 'hist':
            plt.figure(figsize=cnst.DEFAULT_FIGSIZE)
            data_frame[action['col']].hist()
        elif action['kind'] == 'counts':
            plt.figure(figsize=cnst.DEFAULT_FIGSIZE)
            data_frame[action['col']].value_counts(
                'normalize').plot(kind='bar')
        elif action['kind'] == 'density':
            plt.figure(figsize=cnst.DEFAULT_FIGSIZE)
            sns.histplot(data_frame[action['col']], stat='density', kde=True)
        else:
            logging.error("ERROR: Unknown kind of analysis %s", action['kind'])

        plt.savefig(f"{file_path}{action['col']}_{action['kind']}.png")
        plt.savefig(f"{file_path}{action['col']}_{action['kind']}.pdf")
    logging.info("SUCCESS")


def perform_correlation_analysis(
        data_frame,
        file_path=cnst.EDA_IMAGES_PATH):
    '''
    perform correlation anlaysis on data_frame by creating a plot
    input:
            data_frame: pandas dataframe
            file_path: string path to store analysis figures to
    output:
            None
    '''
    logging.info("Performing correlation analysis")
    plt.figure(figsize=cnst.DEFAULT_FIGSIZE)
    sns.heatmap(
        data_frame.corr(),
        annot=False,
        cmap='vlag',
        linewidths=2,
        vmin=-1,
        vmax=1)
    plt.savefig(f"{file_path}correlation_heatmap.png")
    plt.savefig(f"{file_path}correlation_heatmap.pdf")
    logging.info("SUCCESS")

##################### Modeling ############################


def encoder_helper(data_frame, category_lst, response=cnst.TARGET):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
                      [optional argument that could be used for
                      naming variables or index y column]
    output:
            data_frame: pandas dataframe with new columns for
    '''
    logging.info("Encoding categorical features")
    if not isinstance(data_frame, pd.DataFrame):
            logging.error('ERROR: data_frame is not of type pd.DataFrame')
            logging.error('ERROR: data_frame is of type %s', type(data_frame))
            raise TypeError
    if not isinstance(category_lst, list): 
            logging.error('ERROR: category_lst is not of type list')
            logging.error('ERROR: category_lst is of type %s', type(category_lst))
            raise TypeError
    if not response in data_frame.columns:
            logging.error('ERROR: response %s is no data_frame column', response)
            raise KeyError
            
    for category in category_lst:
        category_groups = data_frame.groupby(category)[response].mean()
        data_frame[f'{category}_{response}'] = [category_groups.loc[val]
                                                for val in data_frame[category]]
    logging.info("SUCCESS")
    return data_frame


def perform_feature_engineering(data_frame, response=cnst.TARGET):
    '''
    This function create new features, drops features and returns train and test datasets
    input:
              data_frame: pandas dataframe
              response: string of response name
                        [optional argument that could be used for
                        naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    if not isinstance(data_frame, pd.DataFrame):
            logging.error('ERROR: data_frame is not of type pd.DataFrame')
            logging.error('ERROR: data_frame is of type %s', type(data_frame))
            raise TypeError
    if not response in data_frame.columns:
            logging.error('ERROR: response %s is no data_frame column', response)
            raise KeyError
            
    logging.info("Performing feature engineering")
    data_frame = encoder_helper(data_frame, cnst.CATEGORICAL_FEATURES)
    target = data_frame[response]
    features = pd.DataFrame()
    features[cnst.KEEP_COLUMNS] = data_frame[cnst.KEEP_COLUMNS]
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=cnst.TEST_SIZE, random_state=cnst.SEED)
    logging.info("SUCCESS")
    return features_train, features_test, target_train, target_test


def classification_report_image(models_predictions, output_pth=cnst.MODEL_IMAGES_PATH):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            models_predictions: list containing groud truth and estimators predictons
                model_type: string naming the model that created the predictions
                y_train: training response values
                y_test:  test response values
                log_reg_predictions_train: training predictions from logistic regression
                log_reg_predictions_test: test predictions from logistic regression
                rand_forest_predictions_train: training predictions from random forest
                rand_forest_predictions_test: test predictions from random forest
    output:
             None
    '''
    logging.info("Create classification reports")
    _ = [create_classification_report(prediction, output_pth)
         for prediction in models_predictions]

    logging.info("SUCCESS")


def create_classification_report(prediction,
                                 output_pth=cnst.MODEL_IMAGES_PATH):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            prediction: dictionary containing
                model_type: type of model [Random Forrest, Linear Regression, ...]
                y_train: training response values
                y_test:  test response values
                pred_train: training predictions from estimator
                pred_test: test predictions from estimator
    output:
             None
    '''
    logging.info(
        "Create classification report for %s",
        prediction['model_type'])
    plt.figure()
    plt.rc('figure', figsize=cnst.DEFAULT_FIGSIZE)
    plt.text(0.01, 1.25, str(f"{prediction['model_type']} Train"), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                prediction['y_test'], prediction['pred_test'])), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str(f"{prediction['model_type']} Test"), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                prediction['y_train'], prediction['pred_train'])), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(
        f"{output_pth}{prediction['model_type']}_classification_report.png")
    plt.savefig(
        f"{output_pth}{prediction['model_type']}_classification_report.pdf")
    logging.info("SUCCESS")


def feature_importance_plot(
        model,
        features,
        output_pth=cnst.MODEL_IMAGES_PATH):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            features: pandas dataframe of feature values
            output_pth: path to store the figure
    output:
             None
    '''
    logging.info("Create feature importance plot")
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [features.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=cnst.DEFAULT_FIGSIZE)

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel("Importance")

    # Add bars
    plt.bar(range(features.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(features.shape[1]), names, rotation=90)

    plt.savefig(f"{output_pth}feature_importances.png")
    plt.savefig(f"{output_pth}feature_importances.pdf")
    logging.info("SUCCESS")


def create_roc_plots(
        estimator_list,
        features,
        target,
        output_pth=cnst.MODEL_IMAGES_PATH,
        file_name="ROC_plot"):
    '''
    creates and stores ROC plots for a list of estimators
    input:
            estimator_list: mlist of estimators
            features: pandas dataframe of feature values
            target: pandas dataframe of target values
            output_pth: path to store the figure
            file_name: name to store the figure
    output:
             None
    '''
    logging.info("Create ROC plot")
    plt.figure(figsize=cnst.DEFAULT_FIGSIZE)
    axis = plt.gca()
    for estimator in estimator_list:
        plot_roc_curve(estimator, features, target, ax=axis, alpha=0.8)
    plt.savefig(f"{output_pth}{file_name}.png")
    plt.savefig(f"{output_pth}{file_name}.pdf")
    logging.info("SUCCESS")


def calculate_shap_values(
        estimator,
        features,
        output_pth=cnst.MODEL_IMAGES_PATH):
    '''
    creates and stores a plot of shap values for model explainability
    input:
            estimator: estimators
            features: pandas dataframe of feature values
            output_pth: path to store the figure
    output:
             None
    '''
    logging.info("Create shap values")
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(features)
    plt.figure(figsize=cnst.DEFAULT_FIGSIZE)
    shap.summary_plot(shap_values, features, plot_type="bar")
    plt.savefig(f"{output_pth}shap_values.png")
    plt.savefig(f"{output_pth}shap_values.pdf")
    logging.info("SUCCESS")


def train_models(train_features, test_features, train_targets, test_targets):
    '''
    train, store model results: images + scores, and store models
    input:
              train_features: X training data
              test_features: X testing data
              train_targets: y training data
              test_targets: y testing data
    output:
              None
    '''
    logging.info("Train models")
    
    if train_features.shape[0] != train_targets.shape[0]:
        logging.error('ERROR: Found inconsistent numbers of samples: [%i, %i] in train set', train_features.shape[0], train_targets.shape[0])
        raise ValueError
    if test_features.shape[0] != test_targets.shape[0]:
        logging.error('ERROR: Found inconsistent numbers of samples: [%i, %i] in test set', test_features.shape[0], test_targets.shape[0])
        raise ValueError
    try:
        rfc = train_random_forrest_clf(train_features, train_targets)
        y_train_preds_rf, y_test_preds_rf = create_train_test_predictions(
            rfc, train_features, test_features)
        #calculate_shap_values(rfc, test_features)
        feature_importance_plot(rfc, train_features, cnst.MODEL_IMAGES_PATH)

        lrc = train_logistic_regression_clf(train_features, train_targets)
        y_train_preds_lr, y_test_preds_lr = create_train_test_predictions(
            lrc, train_features, test_features)

        models_predictions = [
            {'model_type': 'Random Forrest Classifier',
             'y_train': train_targets,
                           'y_test': test_targets,
                           'pred_train': y_train_preds_rf,
                           'pred_test': y_test_preds_rf},
            {'model_type': 'Logistic Regression Classifier',
             'y_train': train_targets,
             'y_test': test_targets,
                           'pred_train': y_train_preds_lr,
                           'pred_test': y_test_preds_lr}
        ]
        classification_report_image(models_predictions, cnst.MODEL_IMAGES_PATH)

        create_roc_plots([rfc, lrc], train_features, train_targets,  cnst.MODEL_IMAGES_PATH, "ROC_on_train")
        create_roc_plots([rfc, lrc], test_features, test_targets,  cnst.MODEL_IMAGES_PATH, "ROC_on_test")

        # save best model
        joblib.dump(rfc, f'{cnst.MODEL_PATH}rfc_model.pkl')
        joblib.dump(lrc, f'{cnst.MODEL_PATH}logistic_model.pkl')
        logging.info("SUCCESS")
    except KeyError as err:
        logging.error("ERROR: %s", err)
        raise err

def train_random_forrest_clf(train_features, train_targets):
    '''
    Do a grid-search and train a Random Forrest Classifer
    input:
              train_features: X training data
              train_targets: y training data
    output:
              cv_rfc.best_estimator_: best performing Random Forrest Classifier
    '''
    logging.info("Train Random Forrest Classifier")
    rfc = RandomForestClassifier(random_state=cnst.SEED)
    logging.info("Performing Grid-Search")
    cv_rfc = GridSearchCV(
        estimator=rfc,
        param_grid=cnst.PARAM_GRID,
        cv=cnst.CV)
    cv_rfc.fit(train_features, train_targets)
    logging.info("SUCCESS")
    return cv_rfc.best_estimator_


def train_logistic_regression_clf(train_features, train_targets):
    '''
    Train a Logistic Regression Classifier
    input:
              train_features: X training data
              train_targets: y training data
    output:
              lrc: trained logistic regression classifier
    '''
    logging.info("Train Linear Regression Classifier")
    lrc = LogisticRegression(
        solver=cnst.SOLVER,
        max_iter=cnst.MAX_ITER)
    lrc.fit(train_features, train_targets)
    logging.info("SUCCESS")
    return lrc


def create_train_test_predictions(model, train_features, test_features):
    '''
    Creates predictions for training and test data for a given model
    input:
              model: model used to create predictions
              train_features: X training data
              test_features: X testing data
    output:
              predictions_on_train: prediction of y on training data
              predictions_on_test: prediction of y on testing data
    '''
    logging.info("Create prediction on train and test set")
    predictions_on_train = model.predict(train_features)
    predictions_on_test = model.predict(test_features)
    logging.info("SUCCESS")
    return predictions_on_train, predictions_on_test


if __name__ == '__main__':
    input_data = import_data("./data/bank_data.csv")
    input_data['Churn'] = input_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    perform_eda(input_data)
    X_train, X_test, y_train, y_test = perform_feature_engineering(input_data)
    train_models(X_train, X_test, y_train, y_test)
