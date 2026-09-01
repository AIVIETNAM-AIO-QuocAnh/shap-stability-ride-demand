import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.utilities import load_config, resolve_path

cfg = load_config()

class TrainTest :
    def __init__(self, train_data, test_data, model_key):
        self.X_train, self.y_train = train_data
        self.X_test, self.y_test = test_data
        self.model_key = model_key

    def run(self, fold):
        print(self.model_key)


        #baseline models for testing 
        params = cfg["models"][self.model_key]
        if(self.model_key == "xgboost"):
            self.model = XGBRegressor(**params)
        else : 
            self.model = LGBMRegressor(**params)

        if fold == "hpo":
            run_hpo = RunHpo(self.X_train, self.y_train, self.X_test, self.y_test, self.model_key)
            best_params = run_hpo.run()
            if(self.model_key == "xgboost"):
                self.model = XGBRegressor(**best_params)
            else :
                self.model = LGBMRegressor(**best_params)
        else :
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)
            

    def log(self, model_name, variant, fold): #đưa các kết quả từ run() ra .csv/.txt
        # Hoàn thiện phần xử lí đường dẫn
        path = resolve_path(cfg, 'results') / variant / fold / model_name
        print(f"Logging results to {path}")
        # lưu model

        # lưu metrics ra csv

        # lưu predictions ra csv cho tập evaluation

        # df = pd.DataFrame(self.results)
        # df.to_csv(
        #     path,
        #     mode="a",                     # append
        #     header=not path.exists(), # chỉ ghi header nếu file chưa tồn tại
        #     index=False,
        # )