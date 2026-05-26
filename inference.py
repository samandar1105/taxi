import joblib
import pandas as pd
import json


class TaxiPricePredictor:

    def __init__(
        self,
        model_path="final_model.pkl",
        metadata_path="final_metadata.json"
    ):

        self.model = joblib.load(model_path)

        with open(metadata_path, encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.feature_names = self.metadata["feature_names"]

        print("✓ Model loaded successfully")

    def predict(self, data):

        if isinstance(data, dict):
            data = pd.DataFrame([data])

        data = data[self.feature_names]

        prediction = self.model.predict(data)

        return {
            "prediction": prediction.tolist()
        }


if __name__ == "__main__":

    predictor = TaxiPricePredictor()

    sample_data = {}

    result = predictor.predict(sample_data)

    print(result)
