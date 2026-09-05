# Data Dictionary - Feature và Correlation Artifacts

Mỗi row đại diện cho một pickup zone tại một target hour. Timestamps được chuẩn hóa theo giờ.

## `feature_table.csv`

| Cột | Kiểu | Ý nghĩa | Quy ước |
|---|---|---|---|
| `pu_location_id` | int | Mã pickup zone | Một trong 50 frozen zones; chưa được one-hot encode |
| `target_datetime` | datetime | Target hour `t` | Reference key, không phải model feature |
| `demand` | int | Số request tại zone ở thời điểm `t` | Target; không âm và có thể bằng 0 |
| `hour` | int (0-23) | Giờ lấy từ `target_datetime` | Base feature |
| `dayofweek` | int (0-6) | Thứ trong tuần, Monday = 0 | Base feature |
| `is_holiday` | int (0/1) | Cờ US federal holiday | Base feature |
| `lag_1` | int | Demand tại `t - 1 hour` | Base feature |
| `lag_24` | int | Demand tại `t - 24 hours` | Base feature dùng cho mọi variant |
| `lag_168` | int | Demand tại `t - 168 hours` | Weekly lag dùng cho A và C |
| `lag_336` | int | Demand tại `t - 336 hours` | Weekly lag dùng cho A |
| `lag_504` | int | Demand tại `t - 504 hours` | Weekly lag dùng cho A |
| `median_lag_3w` | float | Median của ba weekly lags | Aggregated weekly feature dùng cho B và C |

### Các ràng buộc đã đảm bảo

- Các required columns không có missing values.
- `(pu_location_id, target_datetime)` là unique key.
- Dữ liệu bắt đầu từ `2025-01-22`, sau `lag_504` warm-up period.
- Demand bằng 0 được giữ lại như một observation hợp lệ.

## `correlation_by_zone.csv`

Pearson correlation được tính cho từng frozen zone và từng weekly-lag pair trong khoảng
`[2025-01-22 00:00, 2025-08-01 00:00)`.

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `pu_location_id` | int | Mã frozen pickup zone |
| `pair` | str | Một trong ba weekly-lag pairs |
| `pearson_r` | float | Pearson coefficient trong [-1, 1]; có thể là `NaN` nếu input không đổi |
| `n_obs` | int | Số hourly observations của zone |

## `correlation_summary.csv`

Mean và sample standard deviation theo từng pair trên 50 zones.

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `pair` | str | Weekly-lag pair |
| `mean_r` | float | Mean của các zone-level coefficients hợp lệ |
| `std_r` | float | Sample standard deviation của các coefficients hợp lệ |
| `n_valid` | int | Số zones có coefficient xác định |
| `n_zones_total` | int | Tổng số zones được xét; kỳ vọng bằng 50 |
| `n_undefined` | int | Số zones có coefficient không xác định |

## Artifacts liên quan

- `variant_feature_map.json`: weekly-feature contract của A/B/C.
- `frozen/top50_zones_frozen.json`: frozen zone vocabulary và selection record.
- `lag_alignment_examples.csv`: 75 full-panel checks cho alignment `target_datetime - lag`.

## `lag_alignment_examples.csv`

File gồm ba frozen zones, năm target hours đại diện và năm lag values.
Mọi giá trị `matches` phải là `True`; source timestamps giúp audit mà không cần
rebuild pre-warm-up panel.

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `pu_location_id` | int | Mã frozen pickup zone |
| `target_datetime` | datetime | Target hour đang được kiểm tra |
| `lag` | int | Độ dài lag tính theo giờ |
| `source_datetime` | datetime | Source hour kỳ vọng (`target_datetime - lag`) |
| `source_demand` | int | Full-panel demand tại source hour |
| `feature_value` | int | Giá trị `lag_<lag>` được tạo tại target hour |
| `matches` | bool | Exact match giữa source value và feature value |
