# Data Dictionary - feature_table.csv & correlation outputs

Mỗi row = 1 (zone, giờ). Đơn vị thời gian: giờ (hourly).

## 1. `feature_table.csv`
| Cột | Kiểu | Mô tả | Ghi chú |
|---|---|---|---|
| `PULocationID` | int | Mã zone (pickup location), 1 trong 50 zone đã freeze | Chưa one-hot |
| `target_datetime` | datetime | Mốc giờ t, giờ mà `demand` được đo | Chỉ để tham chiếu/debug, không phải feature |
| `demand` | int | **Target** — số request tại zone đó, trong giờ đó | >= 0, có thể = 0 (giờ không có request nào) |
| `hour` | int (0-23) | Giờ trong ngày, suy từ `target_datetime` | Feature nền |
| `dayofweek` | int (0-6) | Thứ trong tuần, 0 = Thứ Hai | Feature nền |
| `is_holiday` | int (0/1) | 1 nếu ngày đó là ngày lễ liên bang Mỹ | Feature nền |
| `lag_1` | int | Demand cách đây 1 giờ | Feature nền |
| `lag_24` | int | Demand cách đây 24 giờ (1 ngày) | Feature nền, dùng cho mọi Variant |
| `lag_168` | int | Demand cách đây 168 giờ (1 tuần) | Weekly lag, dùng ở Variant A/C |
| `lag_336` | int | Demand cách đây 336 giờ (2 tuần) | Weekly lag, dùng ở Variant A |
| `lag_504` | int | Demand cách đây 504 giờ (3 tuần) | Weekly lag, dùng ở Variant A |
| `median_lag_3w` | float | median(`lag_168`, `lag_336`, `lag_504`) | Weekly lag gộp, dùng ở Variant B/C |

### Ràng buộc đã đảm bảo

- Không có missing value ở bất kỳ cột bắt buộc nào.
- Không có row (zone, giờ) bị trùng.
- Dữ liệu bắt đầu từ **2025-01-22** (đã cắt warm-up để đảm bảo `lag_504` luôn có đủ lịch sử).
- `demand = 0` là giá trị hợp lệ (giờ không có request), không phải missing.

## 2. `correlation_by_zone.csv`
 
Pearson correlation giữa 3 weekly lag, tính riêng cho từng zone, chỉ trên giai đoạn 22/01–31/07/2025.
 
| Cột | Kiểu | Mô tả | Ghi chú |
|---|---|---|---|
| `PULocationID` | int | Zone được tính correlation | 1 trong 50 zone đã freeze |
| `pair` | str | Cặp weekly lag, dạng `lag_A_vs_lag_B` | 1 trong 3 giá trị: `lag_168_vs_lag_336`, `lag_168_vs_lag_504`, `lag_336_vs_lag_504` |
| `pearson_r` | float | Hệ số Pearson correlation của riêng zone đó, cặp đó | Trong khoảng [-1, 1]; có thể là `NaN` nếu 1 trong 2 cột gần như hằng số trong giai đoạn đo |
| `n_obs` | int | Số giờ dữ liệu dùng để tính correlation cho zone đó | Dùng để đánh giá độ tin cậy của `pearson_r` |
 
## 3. `correlation_summary.csv`
 
Tổng hợp mean ± SD của `pearson_r` trên 50 zone, cho mỗi cặp lag.
 
| Cột | Kiểu | Mô tả | Ghi chú |
|---|---|---|---|
| `pair` | str | Cặp weekly lag | Cùng 3 giá trị như trên |
| `mean_r` | float | Trung bình `pearson_r` trên các zone **hợp lệ** (đã loại `NaN`) | |
| `std_r` | float | Độ lệch chuẩn `pearson_r` trên các zone hợp lệ | |
| `n_valid` | int | Số zone tính được correlation hợp lệ (không `NaN`) | |
| `n_zones_total` | int | Tổng số zone (kỳ vọng = 50) | |
| `n_undefined` | int | Số zone bị loại vì `pearson_r` là `NaN` |  |
 
## File liên quan
 
- `variant_feature_map.json` — định nghĩa cột feature theo Variant A/B/C.
- `frozen/top50_zones_frozen.json` — vocabulary cố định cho one-hot `PULocationID`.

