# Module Pipeline - training, SHAP và cách reproduce

Pipeline nhận `data/folds/` đã được Data role kiểm tra, rồi thực hiện HPO, training,
prediction, SHAP sampling, artifact validation và canonical aggregation cho nghiên cứu
SHAP stability. Module này không tải raw data và không rebuild processed data.

Protocol dùng chung nằm trong `configs/data.yaml` và `configs/model.yaml`; các loader
trong `src/load_config.py` validate configuration trước khi chạy.

## Pipeline overview

| Stage | Input | Output | Mục đích |
|---|---|---|---|
| SHAP sample | Evaluation rows của mỗi fold | `shap_sample_keys.csv` | Freeze cùng một tập semantic rows cho mọi variant/model. |
| HPO | Variant A, split `hpo` | `best_params.json`, `trials.csv` | Chọn cấu hình bằng MAE với đúng 20 Optuna trials/model. |
| HPO comparison | Split `hpo` | Baseline và tuned artifacts | So sánh model chưa tune với cấu hình đã freeze. |
| Core training | A/B/C × 5 folds × 2 models | Model, predictions, metrics, snapshots | Chạy toàn bộ experiment theo temporal protocol. |
| SHAP | Core model + shared sample keys | Raw SHAP, plots, importance | Đo mean absolute feature importance bằng Tree SHAP. |
| QA/summary | Generated artifacts | `run_matrix.csv`, `results/stats/*.csv` | Chặn artifact thiếu/sai trước khi Analysis đọc kết quả. |

## Luồng Pipeline tổng quát

```text
Data handoff: frozen zones + feature table + temporal folds
        |
        | validate Data QA/QC trước khi chạy model
        v
Semantic SHAP sample keys: 100 rows / zone / fold
        |
        +--> Variant A / HPO split --> 20 Optuna trials / model
        |                              |
        |                              v
        |                       frozen best parameters
        |                              |
        +--> baseline + tuned HPO comparison
        |
        +--> Variant A/B/C × fold1-4/final_test × XGBoost/LightGBM
                                       |
                                       v
                         keyed predictions + model snapshots
                                       |
                                       v
                    TreeExplainer + shared SHAP sample keys
                                       |
                                       v
                 artifact matrix QA + mean/std canonical tables
                                       |
                                       v
                                  Analysis handoff
```

## Nguyên tắc protocol

- Chỉ Variant A trên split `hpo` được tune. Mỗi model có đúng 20 Optuna trials,
  `TPESampler(seed=42)` và MAE làm objective.
- Best parameters của mỗi model được freeze và dùng nguyên vẹn cho Variant A/B/C,
  fold1-4 và `final_test`; không tune lại theo variant hoặc fold.
- Feature dùng cho model không chứa raw `pu_location_id`. Semantic row key là
  `(pu_location_id, target_datetime)`.
- SHAP dùng `TreeExplainer(feature_perturbation="tree_path_dependent")`, chọn 100
  rows cho mỗi frozen zone trong mỗi fold và dùng cùng semantic keys cho mọi variant
  và model.
- `shap_importance.csv` dùng mean absolute SHAP value. Fold1-4 được tổng hợp bằng
  mean và sample standard deviation; December `final_test` được báo cáo riêng.
- WAPE trong `metrics.json` là percentage, không phải ratio. `y_pred.csv` luôn chứa
  semantic keys cùng `y_true` và `y_pred`.
- Artifact writers không overwrite file đã tồn tại. Directory có thể tồn tại và có
  `.gitkeep`, nhưng các artifact cùng tên phải chưa tồn tại trước khi chạy.

## Reproduce từ Data handoff

Chạy từ project root trong environment `shap-stability-ride-demand`. Trước khi chạy,
Data artifacts phải có sẵn và Data QA/QC phải pass:

```bash
python -m src.data.qa_checks
```

Một full run theo đúng thứ tự protocol:

```bash
python run.py all
```

Full run sẽ tạo semantic sample keys, chạy HPO, baseline/tuned comparison, 30 core
runs, SHAP, artifact matrix và canonical summary. Pipeline không tự xóa artifact cũ;
fresh run cần output files sạch theo policy của project.

Optuna hiển thị progress cho 20 trials của từng model; full run có một progress bar
cho 30 core training-plus-SHAP runs. Sampling, QA/QC và summary không thêm progress
bar vì các vòng lặp này ngắn hơn hoặc là bước kiểm tra.

## Chạy từng stage

Các lệnh dưới đây dùng khi cần kiểm tra hoặc reproduce một phần đã được xác định rõ:

```bash
# Generate one fold's shared SHAP sample keys.
python run.py sample-shap fold1

# HPO, baseline, and tuned evaluation for one model.
python run.py hpo xgboost
python run.py baseline-hpo xgboost
python run.py tuned-hpo xgboost

# Train and explain one core combination.
python run.py train-core xgboost A fold1
python run.py shap xgboost A fold1

# Optional when all preceding stages were run individually.
python run.py check-matrix
python run.py summarize
```

`run.py` dùng `argparse` subcommands để tránh import-time training và để mỗi stage
có input rõ ràng. `all` là entry point chính; các subcommands phục vụ focused run,
debugging và handoff verification. Nếu chạy `python run.py all`, không chạy lại
`check-matrix` hoặc `summarize` vì `all` đã thực hiện hai bước này.

## Cấu trúc file và artifact

```text
src/
├── load_config.py                # validated data/model configuration
├── utilities.py                  # validated fold loading and model features
├── pipeline/
│   ├── build_pipeline.py         # stage orchestration
│   ├── run_hpo.py                # Variant A Optuna studies
│   ├── train_test.py             # model construction and prediction metrics
│   ├── run_shap.py               # semantic sampling and SHAP artifacts
│   ├── artifacts.py              # non-overwriting artifact writers/loaders
│   ├── qa_checks.py              # Pipeline artifact matrix QA
│   ├── summarize.py              # approved core aggregation
│   └── README.md
└── test/test_pipeline.py         # focused Pipeline contract tests

data/folds/{fold}/shap_sample_keys.csv
results/hpo/{model}/
├── best_params.json
├── trials.csv
└── run_config.json

results/A/hpo/{model}_baseline/
results/A/hpo/{model}/
└── model.pkl, metrics.json, y_pred.csv, run_config.json

results/{A,B,C}/{fold}/{model}/
├── model.pkl, metrics.json, y_pred.csv, run_config.json
├── shap_values.pkl, shap_bar.png, shap_beeswarm.png
├── shap_importance.csv, shap_weekly_group.json, shap_config.json
└── ...

results/run_matrix.csv
results/stats/
├── hpo_comparison.csv
├── performance_summary.csv, performance_aggregated.csv
├── shap_importance_per_fold.csv, feature_importance_stability.csv
└── weekly_group_per_fold.csv, weekly_group_stability.csv
```

`run_config.json` ghi target, semantic row keys, feature order, split boundaries,
seed, model parameters và package versions. `model.pkl` vẫn là approved model format;
package snapshot giúp nhận biết portability limitation khi load artifact.

## QA/QC và handoff

Pipeline QA/QC được tách khỏi Data QA/QC vì hai role kiểm tra hai loại handoff khác nhau:

```text
src/data/qa_checks.py
  raw schemas, aggregates, frozen zones, panel, lags, folds, temporal leakage

src/pipeline/qa_checks.py
  HPO trials, keyed predictions, metrics, snapshots, SHAP samples, artifact matrix
```

`python run.py all` đã chạy `check-matrix` và `summarize` ở cuối workflow. Nếu các
stage được chạy riêng lẻ, chạy hai lệnh dưới đây một lần sau khi mọi artifact đã đủ:

```bash
python run.py check-matrix
python run.py summarize
```

`check-matrix` phải pass trước khi Analysis đọc `results/stats/`. Core summary chỉ
bao gồm MAE, RMSE, WAPE, correlation, SHAP feature importance và weekly-group importance
được định nghĩa trong protocol; CV, rank, paired comparison, significance tests và các
diagnostic khác là follow-up riêng, không được tự động thêm vào core.

Data role chỉ chuẩn bị/kiểm tra data. Pipeline role không thay đổi raw data, không
tune ngoài Variant A/HPO split và không đưa December `final_test` vào fold1-4 stability
aggregate.
