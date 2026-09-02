import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import json
import pandas as pd

from src.utilities import load_config, resolve_path, calculate_metrics

cfg = load_config()

class TrainTest :
    def __init__(self, train_data, test_data, model_key, params):
        self.X_train, self.y_train = train_data
        self.X_test, self.y_test = test_data
        self.model_key = model_key
        self.params = params

    def run(self, fold):
        if self.model_key == "xgboost":
            self.model = XGBRegressor(**self.params)
        else:
            self.model = LGBMRegressor(**self.params)

        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

        self.metrics = {}
        for metric in ("mae", "rmse", "wape"):
            self.metrics[metric] = calculate_metrics(metric, self.y_test, self.y_pred)
            
    def log(self, model_name, variant, fold): #đưa các kết quả từ run() ra .csv/.txt
        # Hoàn thiện phần xử lí đường dẫn
        path = resolve_path(cfg, 'results') / variant / fold / model_name
        path.mkdir(parents=True, exist_ok=True)
        print(f"Logging results to {path}")
        # lưu model
        joblib.dump(self.model, path/"model.joblib")
        # lưu metrics ra csv
        with open(path / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)
        # lưu predictions ra csv cho tập evaluation
        pd.DataFrame({"y_true": self.y_test, "y_pred": self.y_pred}).to_csv(path / "predictions.csv", index=False)
        # df = pd.DataFrame(self.results)
        # df.to_csv(
        #     path,
        #     mode="a",                     # append
        #     header=not path.exists(), # chỉ ghi header nếu file chưa tồn tại
        #     index=False,
        # )