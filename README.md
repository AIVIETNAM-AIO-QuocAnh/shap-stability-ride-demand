# SHAP Stability under Correlated Time-Series Features

## Tổng quan

Project nghiên cứu độ ổn định của SHAP khi mô hình dự báo nhu cầu gọi xe theo giờ ở NYC dùng các
weekly lag có tương quan. Mục tiêu là so sánh cách biểu diễn weekly lag ảnh hưởng đến prediction và
feature attribution; đây không phải hệ thống dự báo vận hành.

Trong working copy hiện tại, 12 raw Parquet files của NYC TLC HVFHV năm 2025 đã được tải ở
`data/raw/`. Khi reproduce ở máy khác, hãy đặt đủ 12 file này vào cùng đường dẫn trước khi chạy pipeline.

## Khái niệm chính

| Khái niệm | Giải thích |
|---|---|
| `demand` | Số request tại một pickup zone trong một target hour; đây là target cần dự báo. |
| Row key | `(pu_location_id, target_datetime)` xác định duy nhất một modeling row. |
| Weekly lags | `lag_168`, `lag_336`, `lag_504` là demand của cùng zone ở 1, 2, 3 tuần trước. |
| Variant A/B/C | A giữ ba lag riêng; B dùng `median_lag_3w`; C dùng `lag_168` và `median_lag_3w`. |
| Temporal folds | HPO dùng tháng 07; Fold 1-4 đánh giá tháng 08-11; `final_test` là tháng 12. |
| SHAP importance | Mean absolute SHAP trên 5.000 sample rows; weekly-group importance là tổng của weekly features. |

## Thiết kế thí nghiệm

| Hạng mục | Thiết lập |
|---|---|
| Dữ liệu | NYC TLC HVFHV 2025, aggregate theo pickup zone × hour |
| Zones | Top-50 zone, freeze theo tổng demand trong Jan-Jun |
| Models | LightGBM và XGBoost |
| HPO | Chỉ tune Variant A, 20 Optuna trials/model, seed 42 |
| Metrics | MAE, RMSE, WAPE (%) |
| SHAP | `TreeExplainer`, `tree_path_dependent`, 100 rows/zone/fold |

## Tái lập toàn bộ experiment

Chạy từ project root:

```bash
# Tạo hoặc cập nhật environment.
conda env create -f environment.yaml
# Nếu environment đã tồn tại, dùng lệnh này thay thế:
# conda env update -f environment.yaml
conda activate shap-stability-ride-demand

# Chỉ chạy nếu data/raw/ chưa có đủ 12 Parquet files.
python -m src.data.download_raw

# Data preparation và QA.
python -m src.data.freeze_zones
python -m src.data.build_panel
python -m src.data.correlation_analysis
python -m src.data.split_folds
python -m src.data.qa_checks

# HPO, baseline/tuned comparison, 30 core runs, SHAP và artifact QA.
python run.py all

# Bảng, hình và báo cáo tổng hợp.
python -m src.analysis.run_stats

# Pipeline contract tests.
python -m unittest src.test.test_pipeline
```

Các output chính nằm ở `data/processed/`, `data/folds/`, `results/` và
`src/dashboard/zones_50.geojson`.

 **Lưu ý:** Do khác biệt về phần cứng và hệ điều hành, kết quả reproduce có thể có chênh lệch số nhỏ trong phạm vi chấp nhận được.

## Dashboard

Chạy sau khi toàn bộ experiment và analysis đã hoàn tất:

```bash
python -m src.dashboard.build_zone_geojson
streamlit run dashboard_app.py
```

Trang **Forecast** hiển thị forecast/actual/error theo zone và hour. Trang **Experiment** hiển thị
correlation, prediction metrics, SHAP importance và weekly-group importance. Xem thêm
[Dashboard README](src/dashboard/README.md).

## Kết quả và tài liệu

- [Báo cáo kỹ thuật](results/stats/summary.md): số liệu và kết luận chính.
- [Data README](src/data/README.md): raw data, feature table, folds và Data QA.
- [Pipeline README](src/pipeline/README.md): HPO, training, prediction, SHAP và Pipeline QA.
- [Analysis README](src/analysis/README.md): aggregation và report generation.
- [Heatmap README](src/heatmap/README.md): nguồn geometry cho dashboard.

## Đóng góp theo vai trò

| Vai trò | Nhiệm vụ |
|---|---|
| Tech Lead | Giữ research question/protocol, điều phối handoff và review kết luận. |
| AI Data | Xử lý raw data, freeze zones, dựng panel/features và kiểm tra leakage/correlation. |
| AI Pipeline | Xây runner, temporal splits, metrics, SHAP sampling, artifact layout và QA. |
| AI Model | Chạy baseline/HPO, freeze model config và tạo prediction/SHAP cho A/B/C. |
| QA/QC | Audit target, lags, splits, config, metrics, SHAP keys và reproduction evidence. |
