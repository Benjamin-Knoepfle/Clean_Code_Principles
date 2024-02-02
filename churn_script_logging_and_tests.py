
'''
Author: Benjamin Knöpfle
E-Mail: benjamin.knoepfle@t-systems.com
Date: 29.01.2024

This module contains all functions needed to test and optimize the functions for the churn_library
'''

#############################################################
#                                                           #
#                    import libraries                       #
#                                                           #
#############################################################
import os
import logging
import pandas as pd
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering, train_models
import constants as cnst

import argparse

    
# setup globals and parameters for libraries
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename=cnst.LOGFILE,
    level=logging.DEBUG,
    filemode='w',
    format='%(asctime)s, %(name)s - %(levelname)s - %(message)s [%(funcName)s]',
    datefmt='%d.%m.%Y %I:%M:%S')

##############################################################
#                                                            #
#                 test implementation                        #
#                                                            #
##############################################################
def imported_data():
        '''
        Loads input data and creates Churn feature aka the target
        '''
        input_data = import_data("./test/bank_data.csv")
        input_data['Churn'] = input_data['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        return input_data

def purge_folder(test_path):
    '''
    Removes all files from folder. Used to clean up after test runs
    '''
    _ = [os.remove(f"{test_path}{f}") for f in os.listdir(test_path)]


class TestImportDdata():
    '''
    Class encapsulating all test cases concerning the import_data function
    '''

    def test_successful_import(self):
        '''
        Test for the successful file import. Positive case.
        Expected Behavior: DataFrame with columns and rows is created.
        '''
        logging.info("Testing import_data with existing file")
        try:
            imported_data = import_data("./test/bank_data.csv")
        except FileNotFoundError as err:
            logging.error("Testing import_data: The file wasn't found")
            raise err
        try:
            assert imported_data.shape[0] > 0
            assert imported_data.shape[1] > 0
            logging.info("Testing import_data: SUCCESS")
        except AssertionError as err:
            logging.error(
                "Testing import_data: The file doesn't appear to have rows and columns")
            raise err

    def test_empty_file_import(self):
        '''
        Test for importing an empty file. Negative Testcase.
        Expected Behavior: An EmptyDataError is raised
        '''
        logging.info("Testing import_data with empty file")
        try:
            _ = import_data("./test/empty_data.csv")
        except pd.errors.EmptyDataError:
            logging.info("Testing import_data: SUCCESS")
            assert True
        except Exception:
            logging.error(
                "Testing import_data: Throws unexpected exception on reading empty files")
            assert False

    def test_file_import_nonexisting(self):
        '''
        Test for importing an non existing file. Negative Testcase.
        Expected Behavior: An FileNotFoundError is raised
        '''
        logging.info("Testing import_data with nonexisting file")
        try:
            _ = import_data("./test/nonexisting.csv")
        except FileNotFoundError:
            logging.info("Testing import_data: SUCCESS")
            assert True
            return
        logging.error(
            "Testing import_data: Throws unexpected exception on reading empty files")
        assert False


class TestEDA():
    '''
    Class encapsulating all test cases concerning the perform_eda function
    '''

    test_path = './test/eda_results/'

    def test_eda_success(self):
        '''
        test perform eda function for a successful run
        '''
        try:
            # setup test
            expected_files = [
                'initial_analysis.txt',
                'Churn_hist.png',
                'Customer_Age_hist.png',
                'Marital_Status_counts.png',
                'Total_Trans_Ct_density.png',
                'correlation_heatmap.png',
                'Churn_hist.pdf',
                'Customer_Age_hist.pdf',
                'Marital_Status_counts.pdf',
                'Total_Trans_Ct_density.pdf',
                'correlation_heatmap.pdf']

            # perform test_action
            perform_eda(imported_data(), self.test_path)

            # check for failure
            # hack Trigger Exception by dividing by zero
            _ = [
                1 /
                0 for f in os.listdir(
                    self.test_path) if f not in expected_files]
            logging.info("Testing import_data: SUCCESS")
            result = True
        except ZeroDivisionError as err:
            logging.error(
                "Testing perform_eda: Expected files are not created")
            result = False

        # clean folder
        purge_folder(self.test_path)
        return result

    def test_eda_missing_features(self):
        '''
        test perform eda function when expected features are missing
        '''
        try:
            # setup test
            test_frame = imported_data()
            test_frame.drop(['Customer_Age', 'Marital_Status'],
                            axis=1, inplace=True)

            # perform test_action
            perform_eda(test_frame, self.test_path)
            result = False
        except KeyError:
            result = True
        except Exception:
            result = False
            logging.error(
                "Testing import_data: Throws unexpected exception perform eda")
        # clean up
        purge_folder(self.test_path)
        return result

    def test_eda_none_df(self):
        '''
        test perform eda function when None value is passed
        '''
        try:
            # setup test
            result = False

            # execute test-action
            perform_eda(None, self.test_path)
        except TypeError as err:
            logging.info('SUCCESS')
            result = True
        # clean up
        purge_folder(self.test_path)
        return result

    def test_eda_none_existing_target_folder(self):
        '''
        test perform eda function when the given target folder is non-existing
        '''
        try:
            result = False
            perform_eda(imported_data(), ',/non_existing_path/')
        except FileNotFoundError as err:
            result = True
        return result


class TestEncoderHelper():
    '''
    Class encapsulating all test cases concerning the encoder_helper function
    '''

    def test_encoder_helper_success(self):
        '''
        test encoder helper
        '''
        try:
            result = True
            test_frame = imported_data().head(3)[['Gender', 'Education_Level', 'Churn']]
            test_frame.loc[0, 'Churn'] = 1
            test_frame.loc[1, 'Churn'] = 1
            test_frame.loc[2, 'Churn'] = 0
            category_lst = ['Gender', 'Education_Level']

            encoded_frame = encoder_helper(test_frame, category_lst, response=cnst.TARGET)

            if encoded_frame.loc[0:'Gender_Churn'] != 0.5: result = False;
            if encoded_frame.loc[1:'Gender_Churn'] != 1: result = False
            if encoded_frame.loc[2:'Gender_Churn'] != 0.5: result = False
            if encoded_frame.loc[0:'Education_Level_Churn'] != 1: result = False
            if encoded_frame.loc[1:'Education_Level_Churn'] != 0.5: result = False
            if encoded_frame.loc[2:'Education_Level_Churn'] != 0.5: result = False
                
        except Exception:
            result = False            
            
        return False
        
    def test_encoder_helper_none_df(self):
        try:
            result = False
            category_lst = ['Gender', 'Education_Level']

            encoded_frame = encoder_helper(None, category_lst, response=cnst.TARGET)
        except TypeError as err:
            logging.info('SUCCESS')
            result = True
        # clean up
        return result
        
    def test_encoder_helper_catefory_lst_is_not_list(self):
        try:
            result = False
            category_lst = 'This is not a list'

            encoded_frame = encoder_helper(imported_data(), category_lst, response=cnst.TARGET)
        except TypeError as err:
            logging.info('SUCCESS')
            result = True
        # clean up
        return result
        
    def test_encoder_helper_empty_category_list(self):
        try:
            result = False
            category_lst = []

            encoded_frame = encoder_helper(imported_data(), category_lst, response=cnst.TARGET)
        except TypeError as err:
            logging.info('SUCCESS')
            result = True
        # clean up
        return result
        
    def test_encoder_helper_non_existing_column_in_category_list(self):
        try:
            result = False
            category_lst = ['Non_Existing']

            encoded_frame = encoder_helper(imported_data(), category_lst, response=cnst.TARGET)
        except KeyError as err:
            logging.info('SUCCESS')
            result = True
        # clean up
        return result   
        
    def test_encoder_helper_quant_features_in_list(self):
        result = False
        category_lst = ['Gender', 'Education_Level', 'Customer_Age']
        encoded_frame = encoder_helper(imported_data(), category_lst, response=cnst.TARGET)
        return result
        
    def test_encoder_helper_non_existing_response(self):
        try:
            result = False
            category_lst = ['Gender', 'Education_Level']
            encoded_frame = encoder_helper(imported_data(), category_lst, 'Non_Existing')
        except KeyError as err:
            logging.info('SUCCESS')
            result = True
        # clean up
        return result


class TestPerformFeatureEngineering():
    '''
    Class encapsulating all test cases concerning the perform_feature_engineering function
    '''

    def test_perform_feature_engineering_success(self):
        '''
        test perform_feature_engineering
        '''
        # perform_feature_engineering
        try:
            result = True
            test_frame = imported_data().head(10)

            features_train, features_test, target_train, target_test = perform_feature_engineering(data_frame, response=cnst.TARGET)
        except Exception as err:
            print(err)
            
    def test_perform_feature_engineering_Non_df(self):    
        try:
            result = False

            features_train, features_test, target_train, target_test = perform_feature_engineering(None, response=cnst.TARGET)
        except TypeError as err:
            logging.info('SUCCESS')
            result = True
        # clean up
        return result
    
    def test_perform_feature_engineering_non_existing_response(self):
        try:
            result = False
            features_train, features_test, target_train, target_test = perform_feature_engineering(imported_data(), response=cnst.TARGET)
        except KeyError as err:
            logging.info('SUCCESS')
            result = True
        # clean up
        return result

class TestTrainModels():
    '''
    Class encapsulating all test cases concerning the train_models function
    '''
    cnst.MODEL_IMAGES_PATH = './test/results/'
    cnst.MODEL_PATH = './test/models/'
    cnst.CV = 2
    

    def test_train_models_switched_features(self):
        '''
        test train_models
        '''
        try:
            # train_models
            result = True
            test_frame = imported_data().head(10)
            features_train, features_test, target_train, target_test = perform_feature_engineering(test_frame, response=cnst.TARGET)

            train_models(features_test, features_train, target_train, target_test)
        except ValueError as er:
            logging.info('SUCCESS')
            assert True
            return
        assert False
        
    def test_train_models_different_features(self):
        '''
        test train_models
        '''
        try:
            # train_models
            result = True
            test_frame = imported_data().head(10)
            features_train, features_test, target_train, target_test = perform_feature_engineering(test_frame, response=cnst.TARGET)
            features_train.drop(['Education_Level'], axis=1, inplace=True)

            train_models(features_test, features_train, target_train, target_test)
        except KeyError as er:
            logging.info('SUCCESS')
            assert True
            return
        assert False
    
    
    def test_train_models_success(self):
        '''
        test train_models
        '''
        
        # train_models
        result = True
        test_frame = imported_data()
        features_train, features_test, target_train, target_test = perform_feature_engineering(test_frame, response=cnst.TARGET)

        train_models(features_train, features_test, target_train, target_test)
        
        #purge_folder(cnst.MODEL_IMAGES_PATH)
        #purge_folder(cnst.MODEL_PATH)