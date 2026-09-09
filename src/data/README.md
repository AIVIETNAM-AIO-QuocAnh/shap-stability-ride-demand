# Module Data

Module Data chuyển 12 file NYC TLC HVFHV năm 2025 thành hourly demand theo pickup zone, feature table,
correlation evidence và temporal folds cho Pipeline/Model.

Trong working copy hiện tại, raw files nằm ở `data/raw/`. Khi reproduce ở máy khác, hãy đặt đủ 12 file
Parquet vào thư mục này. Protocol và paths dùng chung nằm trong `configs/data.yaml`.

## Dữ liệu và feature

| Tầng | Nội dung |
|---|---|
| Raw trip | Đọc `request_datetime` và `PULocationID` từ từng file tháng. |
| Monthly aggregate | Đếm request theo `(pu_location_id, hour)`. |
| Target/row key | `demand` tại `target_datetime`; key là `(pu_location_id, target_datetime)`. |
| Calendar | `hour`, `dayofweek`, `is_holiday`. |
| Short lags | `lag_1`, `lag_24`. |
| Weekly lags | `lag_168`, `lag_336`, `lag_504`. |
| Aggregated lag | `median_lag_3w` cho Variant B/C. |

Top-50 zones được freeze theo tổng demand trong khoảng `[2025-01-01, 2025-07-01)`. Dense panel giữ cả
giờ có demand bằng 0; các lag chỉ dùng dữ liệu trước target hour; model rows bắt đầu sau warm-up 504 giờ.

## Reproduce Data handoff

Chạy từ project root sau khi environment đã được kích hoạt:

```bash
# Chỉ chạy nếu data/raw/ chưa có đủ 12 file.
python -m src.data.download_raw

python -m src.data.freeze_zones
python -m src.data.build_panel
python -m src.data.correlation_analysis
python -m src.data.split_folds
python -m src.data.qa_checks
```

Các output chính:

- `data/processed/frozen/top50_zones_frozen.json`: frozen Top-50 zone vocabulary.
- `data/processed/feature_table.csv`: panel và features dùng cho modeling.
- `data/processed/variant_feature_map.json`: feature contract của A/B/C.
- `data/processed/correlation_by_zone.csv` và `correlation_summary.csv`: weekly-lag correlation.
- `data/processed/lag_alignment_examples.csv`: 75 checks cho `target_datetime - lag`.
- `data/folds/{hpo,fold1,fold2,fold3,fold4,final_test}/`: train/evaluation inputs.

## Data QA

`python -m src.data.qa_checks` kiểm tra raw schemas, frozen zones, feature keys, missing values, lag
alignment, variant contract và temporal leakage. QA phải pass trước khi chạy Pipeline.

Column definitions chi tiết nằm trong [`data_dictionary.md`](../../data/processed/data_dictionary.md).
