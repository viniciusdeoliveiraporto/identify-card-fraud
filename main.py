from metrics.calculate_metrics import model_metrics
from model.fraud_detection_model import FraudDetectionModel
import unittest


model = FraudDetectionModel()
#model_metrics(model)

if __name__ == "__main__":
    unittest.main(module=None)