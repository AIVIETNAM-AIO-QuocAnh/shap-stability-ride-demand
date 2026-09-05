# Module Pipeline — training, SHAP

Phạm vi: vai trò **AI Pipeline** (runner, config, artifact structure, SHAP sampling) +
**AI Model** (baseline, Optuna, train A/B/C, tính prediction/SHAP metrics) theo `m03-proposal.pdf`
mục 4. Không gồm phần data (`src/data/`, xem `src/data/README.md`).

## Luồng tổng quát

```text
run.py
  │
  ├─ 1) HPO (chỉ Variant A, trên split "hpo")
  │     BuildPipeline("A").run_hpo()
  │       └─ RunHpo: Optuna 20 trials/model (TPESampler seed=42)
  │            → results/hpo/{model}/best_params.json
  │
  ├─ 2) Baseline chưa tune (untuned) trên cùng split "hpo"
  │     BuildPipeline("A").run_fold(fold="hpo", tuned=False)
  │       └─ TrainTest(use_tuned=False) → results/A/hpo/{model}_baseline/
  │            (để so sánh baseline-vs-tuned, proposal mục 3.2)
  │
  └─ 3) Core experiment: 3 variant × 5 fold × 2 model = 30 tổ hợp
        for variant in [A, B, C]:
          for fold in [fold1, fold2, fold3, fold4, final_test]:
            pipeline.run_fold(fold, tuned=True)   → train_test.py
            pipeline.run_shap(fold)               → run_shap.py
```

`BuildPipeline` chỉ giữ 1 `variant`, mọi hàm con (`run_hpo`, `run_fold`, `run_shap`) tự load lại
data qua `load_data(fold, variant)` trong `src/utilities.py` — mỗi lần gọi đọc CSV mới, không cache.

## `build_pipeline.py` — `BuildPipeline`

- `run_hpo()`: load data `fold="hpo"`, chạy `RunHpo` cho từng model trong `["xgboost", "lightgbm"]`.
- `run_fold(fold, tuned=True)`: load data, train qua `TrainTest`, lưu `model.pkl` / `metrics.json` /
  `y_pred.csv` vào `results/{variant}/{fold}/{model hoặc model_baseline}/` (tên thư mục thêm hậu tố
  `_baseline` khi `tuned=False`, để tách khỏi kết quả đã tune).
- `run_shap(fold)`: load lại data (để có `X_test` gốc cho SHAP), chạy `RunShap` cho từng model.

## `run_hpo.py` — `RunHpo` (Optuna)

- `objective(trial)`: build params = `model_cfg["models"][model_key]` (baseline) ghi đè bởi giá trị Optuna
  suggest theo `model_cfg.hpo.search_space.{model_key}` (list → `suggest_categorical`, dict `{low,high,log}`
  → `suggest_float`); train, trả về `MAE` trên validation của split `hpo` — đây là objective để
  Optuna minimize (đúng proposal mục 2.4: tuning chỉ dùng MAE, chỉ Variant A, chỉ split HPO).
- `run()`: `TPESampler(seed=model_cfg["seed"])`, số trial = `model_cfg["hpo"]["n_trials"]` (= 20).
  Lưu `study.best_params` vào `results/hpo/{model_key}/best_params.json`.
- Search space và baseline config đều đọc từ `configs/model.yaml` (`hpo.search_space.*`, `models.{model_key}`),
  không hardcode trong code — khớp bảng search space ở proposal mục 2.4.

## `train_test.py` — `TrainTest`

- Hyperparams = baseline config (`configs/model.yaml → models[model_key]`) + `random_state=seed`; nếu
  `use_tuned=True` thì ghi đè bằng `results/hpo/{model_key}/best_params.json` (best config đã freeze
  từ bước Optuna, dùng nguyên vẹn cho mọi variant/fold — không tune lại, đúng proposal mục 2.4).
- `run(fold)`: fit model, predict, tính `mae/rmse/wape` qua `calculate_metrics()` (`src/utilities.py`,
  công thức khớp proposal mục 2.5). Trả `(model, metrics, y_pred)` cho `BuildPipeline.save_artifacts`.

## `run_shap.py` — `RunShap`

- **`main()` (chạy 1 lần, trước core loop):** với mỗi fold (`fold1-4`, `final_test`), lấy `X_test` của
  Variant A, sample 100 row/zone (tổng ~5.000 row, `np.random.default_rng(seed)`), lưu row index vào
  `data/folds/{fold}/sample_indices.csv`. Chạy bằng `python -m src.pipeline.run_shap`.
- **`RunShap.run()` (gọi trong core loop, mỗi variant/model/fold):** `load_sample_indices()` đọc lại
  đúng `sample_indices.csv` của fold đó — **cùng 1 tập row cho cả A/B/C và cả 2 model trong 1 fold**,
  đúng SHAP sampling protocol proposal mục 2.6 (tách sample ra khỏi core loop để đảm bảo không lệch
  giữa các lần gọi, thay vì mỗi lần resample lại).
- `shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")` — cố định cho cả 2 model.
- `save_shap_artifacts`: lưu `shap_values.pkl` (raw Explanation object — traceable, chưa tổng hợp),
  `shap_bar.png`, `shap_beeswarm.png`.
- `save_importance`: tính $I_{j,f} = \frac{1}{|S_f|}\sum_{i \in S_f} |\phi_{i,j,f}|$ (mean SHAP tuyệt
  đối qua sample) cho từng feature → `shap_importance.csv`; cộng dồn importance của các
  `weekly_features` theo variant (đọc từ `data/processed/variant_feature_map.json`) →
  `shap_weekly_group.json` (`{"weekly_group_importance": float}`) — đúng công thức weekly-group
  proposal mục 2.6.

## Cấu trúc `results/`

```text
results/
├── hpo/{xgboost,lightgbm}/best_params.json
└── {A,B,C}/
    ├── hpo/{xgboost,lightgbm}_baseline/    # baseline chưa tune, so sánh baseline-vs-tuned
    └── {fold1,fold2,fold3,fold4,final_test}/{xgboost,lightgbm}/
        ├── model.pkl
        ├── metrics.json                    # mae, rmse, wape
        ├── y_pred.csv
        ├── shap_values.pkl                 # raw per-sample SHAP (Explanation object)
        ├── shap_bar.png, shap_beeswarm.png
        ├── shap_importance.csv             # feature, importance — đã gộp mean qua sample
        └── shap_weekly_group.json          # {"weekly_group_importance": float}
```

## Cách chạy

```bash
# 1) Sinh sample_indices.csv cho từng fold (chỉ cần 1 lần, hoặc khi đổi seed/zone_sample_size)
python -m src.pipeline.run_shap

# 2) HPO + baseline + core A/B/C × 5 fold × 2 model (train + SHAP)
python run.py
```

Các path được khai báo trong `configs/data.yaml` (data artifacts) và `configs/model.yaml`
(results). `src/configuration.py` resolve chúng relative to the repository root để mọi thành viên
có thể reproduce từ project root.
