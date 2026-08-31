from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

class TrainTest :
    def __init__(self, train_data, test_data, model_key):
        self.X_train, self.y_train = train_data
        self.X_test, self.y_test = test_data
        self.model_key = model_key

    def run(self):
        print(f"Running TrainTest")

    def log(self, model_name, variant, fold): #đưa các kết quả từ run() ra .csv/.txt
        # Hoàn thiện phần xử lí đường dẫn
        path = resolve_path(cfg, 'results') / variant / fold / model_name
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