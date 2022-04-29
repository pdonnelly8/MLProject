import unittest
import predictions

#test predict_classification
class TestPredictClassification(unittest.TestCase):
    #Test function returns 0 for healthy sample
    def test_predict_classification_Healthy(self):
        self.assertEqual(predictions.predict_classification('D:\\pauricdonnellyfinalyearproject\\flaskServer\\4_Healthy_Neutral.wav'), 0)
    
    #Test function returns 1 for stroke sample
    def test_predict_classification_Pathological(self):
        self.assertEqual(predictions.predict_classification('D:\\pauricdonnellyfinalyearproject\\flaskServer\\128_Pathological_Neutral.wav'), 1)

    #Test Function handles incorrect filepath
    def test_predict_classification_FileNotFound(self):
        self.assertEqual(predictions.predict_classification('D:\\pauricdonnellyfinalyearproject\\flaskServer\\wrongRecording.wav'), "Error loading audio. Please try again.")

if __name__ == '__main__':
    unittest.main()