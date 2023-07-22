import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from challenge import app


class TestBatchPipeline(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
    @patch('challenge.model.DelayModel.predict', return_value=[0])
    def test_should_get_predict(self, mock_predict):
        data = {
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N", 
                    "MES": 3
                }
            ]
        }
        # The comment below is replaced by the @path decorator and mock_predict.assert_called_once() at the end of function
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0])) # change this line to the model of chosing
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predict": [0]})
        mock_predict.assert_called_once()  # Assert that the mock method was called

    
    @patch('challenge.model.DelayModel.predict', return_value=[0])
    def test_should_failed_unkown_column_1(self, mock_predict):
        data = {       
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N",
                    "MES": 13  # Invalid MES
                }
            ]
        }
        # The comment below is replaced by the @path decorator and mock_predict.assert_called_once() at the end of function
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0])) # change this line to the model of chosing
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 422) # It should be error 422, not 400, so I changed it
        mock_predict.assert_not_called()  # Assert that the mock method should not be called (because of error)

    @patch('challenge.model.DelayModel.predict', return_value=[0])
    def test_should_failed_unkown_column_2(self, mock_predict):
        # I changed data so we check wrong columns 1 by 1
        data = {        
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "O",  # Invalid TIPOVUELO
                    "MES": 3
                }
            ]
        }
        # The comment below is replaced by the @path decorator and mock_predict.assert_called_once() at the end of function
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0])) # change this line to the model of chosing
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 422) # It should be error 422, not 400, so I changed it
        mock_predict.assert_not_called()  # Assert that the mock method should not be called (because of error)

    @patch('challenge.model.DelayModel.predict', return_value=[0])
    def test_should_failed_unkown_column_3(self, mock_predict):
        # I changed data so we check wrong columns 1 by 1
        data = {        
            "flights": [
                {
                    "OPERA": "Argentinas",  # Invalid OPERA
                    "TIPOVUELO": "N", 
                    "MES": 1
                }
            ]
        }
        # The comment below is replaced by the @path decorator and mock_predict.assert_called_once() at the end of function
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0])) # change this line to the model of chosing
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 422) # It should be error 422, not 400, so I changed it
        mock_predict.assert_not_called()  # Assert that the mock method should not be called (because of error)