# Module Data - pipeline & cách dùng dữ liệu

## Luồng dữ liệu tổng quát

```text
data/raw ──aggregate──▶ data/processed/monthly_agg - freeze zone──▶ top50_zones_frozen.json
(12 file/tháng)          (agg theo zone×hour, thưa)
                                    │
                                    ▼
                          build hourly panel
          dense grid 50 zone × 8760 giờ + lag + calendar feature
                                    │
                                    ▼
                    data/processed/feature_table.csv
                    data/processed/variant_feature_map.json
                                    │
                        ┌───────────┴───────────┐
                        ▼                        ▼
              QA + correlation summary    QA/QC lag alignment
              (correlation_analysis.py)      + split fold
                        │                        │
                        ▼                        ▼
          correlation_by_zone.csv,          data/folds/
          correlation_summary.csv,      (hpo, fold1-4, final_test)
          data_dictionary.md
```

Nguyên tắc:

- Split **cố định theo thời gian**, không random K-fold.
- 50 zone đã **freeze** dựa trên demand Jan–Jun, cố định cho toàn bộ experiment.
- 3 Variant A/B/C dùng **chung 1 bộ file**, chỉ khác tập cột weekly-lag được chọn.

## Tiền xử lý: sinh dữ liệu từ raw

Yêu cầu raw data đặt sẵn tại `data/raw/`, đúng tên chuẩn TLC:
```
data/raw/fhvhv_tripdata_2025-01.csv ... fhvhv_tripdata_2025-12.csv
```

Chạy từ **root project**, theo đúng thứ tự:

```bash
python src/data/freeze_zones.py
python src/data/build_panel.py
python src/data/correlation_analysis.py
python src/data/split_folds.py
```

### Bước 1–2: `aggregate_month.py` + `freeze_zones.py`

- Đọc **chỉ 2 cột** `request_datetime`, `PULocationID` từ mỗi file raw.
- Aggregate từng tháng theo `(zone, giờ)` → đếm số request, giải phóng raw
  data ngay sau khi xong 1 tháng.
- Chọn **50 zone có tổng demand cao nhất trong Jan–Jun 2025**, freeze danh
  sách.

Output: `data/processed/monthly_agg/agg_2025-01.csv ... agg_2025-12.csv`,
`data/processed/frozen/top50_zones_frozen.json`.

> Script báo lỗi nếu `top50_zones_frozen.json` đã tồn tại (tránh vô tình ghi
> đè zone đã freeze). Xoá file thủ công nếu thật sự cần chọn lại.

### Bước 3: `build_panel.py`

- Dựng **dense grid**: 50 zone × mọi giờ trong năm 2025 (kể cả giờ demand = 0).
- Tính feature thời gian: `hour` (0–23), `dayofweek` (0=Thứ 2), `is_holiday`.
- Tính lag: `lag_1`, `lag_24`, `lag_168`, `lag_336`, `lag_504` (groupby zone + shift).
- Tính `median_lag_3w` = median(`lag_168`, `lag_336`, `lag_504`) cho Variant B/C.
- Cắt bỏ ~21 ngày đầu năm (warm-up, thiếu đủ lịch sử cho `lag_504`) → dữ liệu
  còn lại bắt đầu từ **22/01/2025**.

Output: `data/processed/feature_table.csv`, `data/processed/variant_feature_map.json`.

### Bước 4: `correlation_analysis.py`

- **QA**: kiểm tra missing value, duplicate `(zone, giờ)`, tỷ lệ zero-demand
  (cảnh báo zone nào >95% giờ = 0).
- Tính **Pearson correlation** giữa 3 weekly lag (`lag_168`, `lag_336`,
  `lag_504`), riêng cho từng zone, trên giai đoạn cố định 22/01–31/07/2025
  (mục 2.2 proposal). Tổng hợp mean ± SD trên 50 zone cho mỗi cặp.
- Sinh **data dictionary** mô tả toàn bộ cột của `feature_table.csv` và 2
  file correlation, để bàn giao cho role Pipeline/Model.

Output: `data/processed/correlation_by_zone.csv`,
`data/processed/correlation_summary.csv`, `data/processed/data_dictionary.md`.

### Bước 5: `split_folds.py`

- **QA/QC bắt buộc**: tự tra ngược lag trong chính feature table
  để xác minh không lệch alignment.
- Split theo thời gian thành 6 giai đoạn: HPO, Fold 1–4, Final test — đúng
  bảng ngày mục 2.3 của proposal (train luôn bắt đầu 22/01, mở rộng dần;
  eval là 1 tháng liền kề ngay sau train, không có gap).

Output: `data/folds/{hpo,fold1,fold2,fold3,fold4,final_test}/`.

## Cấu trúc file

```text
data/
├── raw/                            # 12 file gốc
├── processed/
│   ├── monthly_agg/                # 12 file aggregate thưa theo tháng
│   ├── feature_table.csv           # bảng đầy đủ trước khi split
│   ├── variant_feature_map.json    # định nghĩa cột feature theo Variant A/B/C
│   ├── correlation_by_zone.csv     # Pearson correlation weekly lag, theo từng zone
│   ├── correlation_summary.csv     # mean ± SD correlation trên 50 zone
│   ├── data_dictionary.md          # mô tả cột feature_table + correlation output
│   └── frozen/
│       └── top50_zones_frozen.json # danh sách 50 zone cố định
└── folds/
    ├── hpo/         train.csv, val.csv
    ├── fold1/       train.csv, val.csv
    ├── fold2/       train.csv, val.csv
    ├── fold3/       train.csv, val.csv
    ├── fold4/       train.csv, val.csv
    └── final_test/  train.csv, test.csv
```

## Dùng: lấy dữ liệu cho training

### Các cột trong mỗi file

| Cột | Ý nghĩa |
|---|---|
| `PULocationID` | zone id |
| `target_datetime` | mốc giờ |
| `demand` | **target** |
| `hour`, `dayofweek`, `is_holiday` | feature nền |
| `lag_1`, `lag_24`, `lag_168`, `lag_336`, `lag_504`, `median_lag_3w` | feature lag, chọn theo Variant |

Mô tả đầy đủ + ràng buộc đã QA: xem `data/processed/data_dictionary.md`.

### Chọn feature theo Variant

```python
import json

with open("data/processed/variant_feature_map.json") as f:
    variant_map = json.load(f)

base = variant_map["base_features"]  # ["hour","dayofweek","is_holiday","lag_1","lag_24"]

variant = "A"  # hoặc "B", "C"
weekly = variant_map["variants"][variant]["weekly_features"]

feature_cols = base + weekly
```

### One-hot `PULocationID`

```python
import pandas as pd

with open("data/processed/frozen/top50_zones_frozen.json") as f:
    zone_ids = json.load(f)["zone_ids"]

def add_zone_onehot(df, zone_ids):
    df["PULocationID"] = pd.Categorical(df["PULocationID"], categories=zone_ids)
    dummies = pd.get_dummies(df["PULocationID"], prefix="zone")
    return pd.concat([df, dummies], axis=1), dummies.columns.tolist()
```

### Train theo fold

```python
train = pd.read_csv("data/folds/fold1/train.csv", parse_dates=["target_datetime"])
val   = pd.read_csv("data/folds/fold1/val.csv",   parse_dates=["target_datetime"])

train, zone_cols = add_zone_onehot(train, zone_ids)
val, _           = add_zone_onehot(val, zone_ids)

X_train, y_train = train[feature_cols + zone_cols], train["demand"]
X_val,   y_val   = val[feature_cols + zone_cols],   val["demand"]
```

- `hpo`: dùng để tune hyperparameter.
- `fold1`–`fold4`: train/validation cho cross-validation theo thời gian.
- `final_test`: dùng để đánh giá cuối cùng, không dùng để chọn model/tune.