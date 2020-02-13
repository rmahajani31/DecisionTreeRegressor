import unittest
from DecisionTreeRegressor import *
from Encoder import *

class TestDecisionTreeClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_file = "test_encoder.csv"
        cls.dt_classifier = DecisionTreeRegressor(cls.test_file, 1)
        cls.en = Encoder(cls.dt_classifier.data, ["Gender", "Color", "Sport", "Dominant Hand", "Home State", "Allergy", "Food"])
        #cls.dt_classifier.data = cls.en.encode()

    def test_train(self):
        self.dt_classifier.train()
        self.assertAlmostEqual(self.dt_classifier.predict(["Male", "Yellow", "Football", "Left Handed", "Maryland", "Allergies", 5.43]), 4.635)
        self.assertAlmostEqual(self.dt_classifier.predict(["Male", "Yellow", "Football", "Left Handed", "Maryland", "Allergies", 6.8]), 2.946666667)


