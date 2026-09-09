# Module Pipeline

Pipeline nhận Data handoff từ `data/folds/` và thực hiện HPO, training, prediction, SHAP, artifact QA
và summary tables. Pipeline không tải raw data hoặc dựng lại Data handoff.

Model/SHAP settings nằm trong `configs/model.yaml`; data paths và temporal splits nằm trong
`configs/data.yaml`.

## Protocol

- Tune Variant A trên split `hpo`, đúng 20 Optuna trials/model, seed 42 và objective MAE.
- Freeze best parameters của mỗi model rồi dùng cho A/B/C, Fold 1-4 và `final_test`.
- Models: LightGBM và XGBoost. Variants: A/B/C.
- Row key của prediction/SHAP là `(pu_location_id, target_datetime)`.
- SHAP dùng `TreeExplainer(feature_perturbation="tree_path_dependent")`, 100 rows/zone/fold,
  dùng chung sample keys giữa variants và models.
- `shap_importance.csv` là mean absolute SHAP; Fold 1-4 được tổng hợp bằng mean và sample standard
  deviation; December `final_test` báo cáo riêng.

## Reproduce toàn bộ Pipeline

Data QA phải pass trước:

```bash
python -m src.data.qa_checks
python run.py all
```

`python run.py all` thực hiện sample-key generation, HPO, baseline/tuned comparison, 30 core runs,
SHAP, artifact matrix QA và canonical summary tables.

## Reproduce một core run

Ví dụ Variant A / Fold 1 / LightGBM:

```bash
python run.py sample-shap fold1
python run.py hpo lightgbm
python run.py baseline-hpo lightgbm
python run.py tuned-hpo lightgbm
python run.py train-core lightgbm A fold1
python run.py shap lightgbm A fold1
```

Core artifacts được ghi vào `results/A/fold1/lightgbm/`, gồm `model.pkl`, `y_pred.csv`, `metrics.json`,
`run_config.json`, `shap_values.pkl`, `shap_importance.csv` và `shap_weekly_group.json`.

Khi đủ toàn bộ ma trận, chạy thêm:

```bash
python run.py check-matrix
python run.py summarize
```

## QA và handoff

Pipeline QA kiểm tra HPO trials, keyed predictions, metrics, run configurations, SHAP sample rows và
đủ 30 tổ hợp core. Analysis đọc các artifact sau khi Pipeline hoàn tất.

Chi tiết bảng tổng hợp nằm trong [`results/stats/summary.md`](../../results/stats/summary.md).
