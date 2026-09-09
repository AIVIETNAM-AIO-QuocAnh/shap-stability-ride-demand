# Module Dashboard

Dashboard là interactive demo bằng Streamlit để xem prediction theo zone/hour và kết quả SHAP của
experiment. Dashboard chỉ đọc artifact đã sinh, không train lại model.

## Chuẩn bị

Chạy Data, Pipeline và Analysis trước:

```bash
python -m src.data.qa_checks
python run.py all
python -m src.analysis.run_stats
```

## Khởi động

Từ project root:

```bash
python -m src.dashboard.build_zone_geojson
streamlit run dashboard_app.py
```

Dependency của dashboard (`streamlit`, `plotly`, `pyshp`, `pyproj`) nằm trong `environment.yaml`.

## Các trang

| Trang | Nội dung |
|---|---|
| **Forecast** | Bản đồ forecast, actual và error; bảng Top N zone; profile demand 24 giờ; weekly-lag lines. |
| **Experiment** | Weekly-lag correlation, prediction metrics, SHAP feature/weekly-group importance và HPO comparison. |

## Nguồn dữ liệu

- Prediction: `results/{A,B,C}/{fold1..fold4,final_test}/{xgboost,lightgbm}/y_pred.csv`.
- Metrics và SHAP summary: `results/stats/`.
- Bản đồ: `src/dashboard/zones_50.geojson`, được build từ shapefile trong `src/heatmap/taxi_zones/`.
- Ngày lễ: cùng `USFederalHolidayCalendar` được dùng khi tạo feature `is_holiday`.

Ảnh minh họa dashboard nằm ở [`assets/dashboard-forecast.png`](../../assets/dashboard-forecast.png).
