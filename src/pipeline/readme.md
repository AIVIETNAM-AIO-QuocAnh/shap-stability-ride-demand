# HPO flow

HPO = tìm bộ tham số tối ưu cho model trên tập `hpo`.

## Luồng chạy

```
setup thí nghiệm ở `run.py` (Không phải cái trong src.test)
buildpipeline: giữ variant, và cung cấp data cho các tác vụ con
các phase của thí nghiệm : run_hpo -> run_fold -> run_shap(đang implement) -> run_stats(bổ sung sau) -> ... 
```

Quy trình:

1. Load dữ liệu `hpo` train/val
2. Chạy Optuna cho từng model: `xgboost`, `lightgbm`
3. Tìm `best_params`
4. Lưu vào `results/hpo/`

## Đầu vào HPO

- dữ liệu train `data/folds/hpo/train.csv`
- dữ liệu val `data/folds/hpo/val.csv`
- feature mapping theo variant
- search space trong `config.yaml`

## Đầu ra HPO

- file JSON chứa tham số tốt nhất, ví dụ:
  - `results/hpo/xgboost_best_params.json`
  - `results/hpo/lightgbm_best_params.json`

Nội dung mẫu:

```json
{
  "learning_rate": 0.05,
  "max_depth": 6,
  "n_estimators": 300
}
```

## Mục đích

Tên gọi: tìm bộ siêu tham số tốt nhất cho các model trước khi đi vào bước freeze và dùng lại ở các fold/final sau.