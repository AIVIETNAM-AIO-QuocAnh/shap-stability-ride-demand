# Module Data - pipeline và cách dùng dữ liệu

Data role chuyển 12 file NYC TLC HVFHV năm 2025 thành hourly demand theo pickup
zone, feature table, correlation evidence và temporal folds cho Pipeline/Model.
README này mô tả ngắn gọn dữ liệu, lý do xử lý và cách reproduce.

Protocol và paths dùng chung nằm trong `configs/data.yaml`; model/HPO/SHAP settings
nằm trong `configs/model.yaml`. `src/configuration.py` đọc mỗi file một lần và
kiểm tra các field cần thiết trước khi pipeline chạy.

## Dataset overview

Mỗi raw file là một tháng, dạng Parquet:
`fhvhv_tripdata_2025-01.parquet` ... `fhvhv_tripdata_2025-12.parquet`.

| Layer | Fields | Mục đích |
|---|---|---|
| Raw trip | `request_datetime`, `PULocationID` | Xác định thời điểm request và pickup zone. Đây là hai raw fields duy nhất được dùng. |
| Monthly aggregate | `pu_location_id`, `hour`, `trip_count` | Đếm số request theo zone và hour, giảm kích thước dữ liệu cho các bước sau. |
| Target/row key | `pu_location_id`, `target_datetime`, `demand` | Một modeling row = demand của một zone tại một target hour. |
| Calendar features | `hour`, `dayofweek`, `is_holiday` | Mô tả pattern theo giờ, ngày trong tuần và holiday. |
| Short-term lags | `lag_1`, `lag_24` | Bổ sung demand gần đây của cùng zone. |
| Weekly lags | `lag_168`, `lag_336`, `lag_504` | Nhóm feature chính để nghiên cứu correlation và SHAP stability. |
| Aggregated weekly lag | `median_lag_3w` | Đại diện gộp của ba weekly lags cho Variant B/C. |

Top-50 zone được chọn theo tổng `trip_count` trong khoảng chính xác
`[2025-01-01 00:00, 2025-07-01 00:00)`. Selection đọc đủ 12 monthly aggregates
rồi mới filter theo `hour`, vì filename month là pickup-time partition và
`request_datetime` có thể nằm ngoài tháng ghi trên filename.

## Luồng dữ liệu tổng quát

```text
12 raw TLC Parquet files (2025-01 ... 2025-12)
        |
        | read request_datetime + PULocationID only
        v
Monthly zone-hour aggregates
        |
        | concatenate all 12 files
        | filter hour in [2025-01-01, 2025-07-01)
        v
Deterministic frozen Top-50 zones
        |
        | keep frozen zones and build dense hourly grid
        v
Dense panel: 50 zones x 8,760 hours
        |
        | fill missing demand = 0
        | add calendar features and past-only lags
        | drop 504-hour warm-up
        v
feature_table.csv + variant map + lag examples
        |
        +--> weekly-lag correlation artifacts
        |
        +--> HPO/Fold 1-4/final-test inputs
        |
        v
Pipeline / Model / Analysis / QA-QC handoff
```

Nguyên tắc chính:

- `Top-50` được freeze một lần và dùng chung cho toàn experiment.
- Dense panel giữ cả những hour có `demand = 0`.
- Lags chỉ dùng demand của cùng zone ở các hour trước đó.
- Model rows bắt đầu từ `2025-01-22` sau khi loại `lag_504` warm-up.
- Variant A/B/C dùng chung table; chỉ khác weekly feature columns.

## Tiền xử lý: reproduce từ raw

Chạy từ project root, trong environment `shap-stability-ride-demand`:

Trước khi chạy, kiểm tra `configs/data.yaml` để xác nhận paths và protocol; không
cần sửa các script Data để đổi dataset settings.

```bash
python -m src.data.download_raw
python -m src.data.freeze_zones
python -m src.data.build_panel
python -m src.data.correlation_analysis
python -m src.data.split_folds
python -m src.data.qa_checks
```

`download_raw.py` là optional nếu 12 raw files đã có trong `data/raw/`.

### Bước 1: download, aggregate và freeze zone

- `download_raw.py` tải official Parquet files và kiểm tra hai required columns.
- `aggregate_month.py` được gọi cho từng tháng: đọc hai columns, floor
  `request_datetime` về hour và đếm request theo `(pu_location_id, hour)`.
- `freeze_zones.py` nối đủ 12 aggregates, filter khoảng Jan-Jun chính xác,
  sort theo demand giảm dần và `pu_location_id` tăng dần.

Outputs: `data/processed/monthly_agg/agg_2025-01.csv` ... `agg_2025-12.csv`
và `data/processed/frozen/top50_zones_frozen.json`.

`top50_zones_frozen.json` là deterministic checkpoint: exact match sẽ là
`no-op`; mismatch là hard error. Chỉ remove và regenerate sau khi có approved
input hoặc protocol change.

### Bước 2: build panel và features

`build_panel.py` giữ 50 frozen zones, tạo dense grid cho toàn bộ năm 2025,
fill missing demand bằng 0, thêm calendar features và group-wise lags. Sau đó
tạo `median_lag_3w`, one-hot vocabulary helper và loại 504 giờ warm-up.

Outputs: `feature_table.csv`, `variant_feature_map.json` và
`lag_alignment_examples.csv`.

### Bước 3-5: correlation, folds và QA

- `correlation_analysis.py`: tính Pearson correlation theo zone cho ba weekly
  lag pairs trên `2025-01-22` đến trước `2025-08-01`, đồng thời sinh
  `data_dictionary.md`.
- `split_folds.py`: kiểm tra lag alignment rồi ghi HPO, Fold 1-4 và
  `final_test` theo các time boundaries đã khóa.
- `qa_checks.py`: chạy independent audit cho toàn bộ Data handoff.

## Cấu trúc file

```text
src/
├── configuration.py                   # shared validated config loaders
└── data/
    ├── download_raw.py
    ├── aggregate_month.py
    ├── freeze_zones.py
    ├── build_panel.py
    ├── correlation_analysis.py
    ├── split_folds.py
    └── qa_checks.py

configs/
├── data.yaml                         # data paths và locked data protocol
└── model.yaml                        # model, HPO và SHAP settings

data/
├── raw/                              # 12 source Parquet files
├── processed/
│   ├── monthly_agg/                  # sparse zone-hour aggregates
│   ├── frozen/top50_zones_frozen.json
│   ├── feature_table.csv
│   ├── variant_feature_map.json
│   ├── lag_alignment_examples.csv
│   ├── correlation_by_zone.csv
│   ├── correlation_summary.csv
│   └── data_dictionary.md
└── folds/
    ├── hpo/                          # train.csv + val.csv
    ├── fold1/ ... fold4/             # train.csv + val.csv
    └── final_test/                   # train.csv + test.csv
```

Data bàn giao:

- Pipeline/Model dùng `feature_table.csv`, frozen zones, `variant_feature_map.json`
  và `data/folds/`.
- Analysis dùng `correlation_by_zone.csv` và `correlation_summary.csv`.
- QA/QC dùng toàn bộ artifacts, đặc biệt `lag_alignment_examples.csv`.
- Column definitions chi tiết nằm trong `data/processed/data_dictionary.md`.

## QA/QC và cách dùng cho training

Chạy final audit:

```bash
python -m src.data.qa_checks
```

QA/QC kiểm tra:

1. đủ 12 raw schemas và monthly aggregate keys hợp lệ;
2. frozen Top-50 schema, thứ tự zone và `zone_total_demand` khớp independent
   Jan-Jun recomputation;
3. feature table có 50 zones, hourly keys duy nhất, demand/lags hợp lệ và
   warm-up bắt đầu đúng `2025-01-22`;
4. Variant A/B/C đúng feature contract và không có feature ngoài scope;
5. 75 lag examples có `source_datetime = target_datetime - lag`,
   `source_demand = feature_value` và `matches = True`;
6. mọi temporal split không overlap và December chỉ nằm trong `final_test`.

Manual spot check: mở `lag_alignment_examples.csv`, chọn một row, trừ `lag`
hours khỏi `target_datetime`, rồi đối chiếu `source_datetime`, `source_demand`
và `feature_value`. `split_folds.py` cũng thực hiện check này trước khi ghi
fold files.

Pipeline/Model có thể load một variant như sau:

```python
from src.utilities import load_data

X_train, y_train, X_val, y_val = load_data("fold1", "A")
```

Data role chỉ chuẩn bị và kiểm tra dữ liệu; không tune model, chạy HPO hoặc
tạo prediction/SHAP results.
