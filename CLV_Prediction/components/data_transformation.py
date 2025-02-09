import sys

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from CLV_Prediction.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from CLV_Prediction.entity.config_entity import DataTransformationConfig, DataValidationConfig
from CLV_Prediction.components.data_validation import DataValidation
from CLV_Prediction.components.data_ingestion import DataIngestion
from CLV_Prediction.entity.artifact_entity import DataTransformationArtifact
from CLV_Prediction.exception import CLVexception
from CLV_Prediction.logger import logging
from CLV_Prediction.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.datavalidator = DataValidation(DataValidationConfig)
            self.dataingester = DataIngestion()
            self.data_ingestion_artifact = self.dataingester.initiate_data_ingestion()
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CLVexception(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CLVexception(e, sys)
        
    def manual_encoding(self, df):
        df['entitlement_period'] = df['entitlement_period'].replace({'Annual':12,'Monthly':1,'Bi-annual':24})
        df['hourspendonapp'] = df['hourspendonapp'].replace({'Medium':3,'High':5,'Low':1})
    
        return df 


    
    def get_data_transformer_object(self) -> Pipeline:
        """
        Method Name :   get_data_transformer_object
        Description :   This method creates and returns a data transformer object for the data
        
        Output      :   data transformer object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )

        try:
            logging.info("Got numerical cols from schema config")

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            ordinal_encoder = OrdinalEncoder()

            logging.info("Initialized StandardScaler, OneHotEncoder, OrdinalEncoder")

            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']
            num_features = self._schema_config['num_features']

            logging.info("Initialize PowerTransformer")

            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns),
                    ("StandardScaler", numeric_transformer, num_features)
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )
            return preprocessor

        except Exception as e:
            raise CLVexception(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline 
        
        Output      :   data transformer steps are performed and preprocessor object is created  
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if  self.datavalidator.initiate_data_validation().validation_status:
                logging.info("Starting data transformation")
                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")

                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                logging.info("Got train features and test features of Training dataset")

                input_feature_train_df = self.manual_encoding(input_feature_train_df)

                logging.info("transformed entitlement period and hourspendonapp")

                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)

                target_feature_test_df = test_df[TARGET_COLUMN]

                input_feature_test_df = self.manual_encoding(input_feature_test_df)

                logging.info("transformed entitlement period and hourspendonapp fir test")

                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)

                logging.info(
                    "Used the preprocessor object to fit transform the train features"
                )

                input_feature_test_arr = preprocessor.transform(input_feature_test_df)

                train_arr = np.c_[
                    input_feature_train_arr, np.array(target_feature_train_df)
                ]

                test_arr = np.c_[
                    input_feature_test_arr, np.array(target_feature_test_df)
                ]

                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

                logging.info("Saved the preprocessor object")

                logging.info(
                    "Exited initiate_data_transformation method of Data_Transformation class"
                )

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
                return data_transformation_artifact
            else:
                raise Exception(self.datavalidator.initiate_data_validation().message)

        except Exception as e:
            raise CLVexception(e, sys) from e