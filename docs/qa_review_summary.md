# Báo cáo tổng hợp QA Review

**Project:** *SHAP Stability under Correlated Time-Series Features for Hourly Zone-Level Ride-Demand Forecasting*  
**Dataset:** NYC TLC HVFHV Trip Records, Jan–Dec 2025  
**Models:** LightGBM, XGBoost  
**Variants:** A, B, C  
**Evaluation periods:** Fold 1–4 và December final test  
**Core runs được review:** 30

## 1. Mục tiêu review

Sau khi core experiment hoàn tất, nhóm thực hiện một vòng QA review độc lập nhằm kiểm tra tính nhất quán giữa data pipeline, experimental protocol và các kết quả được sử dụng trong technical report. Mục tiêu của bước này không phải tạo thêm experiment mới, mà là xác nhận rằng các kết quả đã lưu có thể được đối chiếu lại từ những artifact đầu ra của pipeline.

Review gồm hai phần chính. Trước hết, phần data và protocol tập trung vào frozen top-50 zones, feature contract của A/B/C, lag alignment, temporal boundaries, việc tách riêng December final test và SHAP sampling protocol. Tiếp theo, phần result verification tính lại prediction metrics và SHAP feature importance từ các artifact đầu ra, sau đó so sánh với kết quả đã lưu.

Nhìn chung, các kiểm tra không phát hiện sai lệch làm thay đổi experimental protocol hoặc các kết luận chính của nghiên cứu.

## 2. Kiểm tra lại prediction metrics

Đối với mỗi core run, MAE, RMSE và WAPE được tính lại từ `y_pred.csv` rồi đối chiếu với các giá trị đã lưu trong `metrics.json`. Việc so sánh sử dụng tolerance phù hợp với sai số floating-point.

**Kết quả: 30/30 core runs PASS.**

| Variant | Model | PASS / Runs |
|---|---|---:|
| A | LightGBM | 5/5 |
| A | XGBoost | 5/5 |
| B | LightGBM | 5/5 |
| B | XGBoost | 5/5 |
| C | LightGBM | 5/5 |
| C | XGBoost | 5/5 |

Mức chênh lệch tuyệt đối lớn nhất giữa giá trị tính lại và giá trị đã lưu được trình bày dưới đây.

| Model | Max abs diff — MAE | Max abs diff — RMSE | Max abs diff — WAPE |
|---|---:|---:|---:|
| LightGBM | 3.55e-15 | 0 | 1.78e-15 |
| XGBoost | 4.92e-08 | 1.60e-07 | 1.89e-08 |

Các sai lệch đều nằm ở mức numerical precision. Vì vậy, các giá trị MAE, RMSE và WAPE được sử dụng trong phần kết quả có thể được xem là nhất quán với prediction artifacts.

## 3. Kiểm tra lại SHAP importance

SHAP importance được tính lại trực tiếp từ raw `shap_values.pkl` theo định nghĩa:

`mean(abs(SHAP))`

Sau đó, kết quả được căn chỉnh theo **feature name** trước khi đối chiếu với `shap_importance.csv`. Cách căn chỉnh này là cần thiết vì raw SHAP matrix giữ model-column order, trong khi bảng importance có thể được sắp xếp lại theo giá trị importance.

**Kết quả: 30/30 core runs PASS.**

| Variant | Model | PASS / Runs | SHAP samples / run | Model features |
|---|---|---:|---:|---:|
| A | LightGBM | 5/5 | 5,000 | 58 |
| A | XGBoost | 5/5 | 5,000 | 58 |
| B | LightGBM | 5/5 | 5,000 | 56 |
| B | XGBoost | 5/5 | 5,000 | 56 |
| C | LightGBM | 5/5 | 5,000 | 57 |
| C | XGBoost | 5/5 | 5,000 | 57 |

Các kiểm tra bổ sung cho thấy:

- raw SHAP artifact tồn tại ở 30/30 runs;
- SHAP importance artifact tồn tại ở 30/30 runs;
- feature set khớp ở 30/30 runs;
- SHAP importance khớp sau khi căn chỉnh theo feature name ở 30/30 runs;
- mỗi run sử dụng đúng 5,000 SHAP rows;
- số model features lần lượt là A = 58, B = 56 và C = 57.

`feature_order_match = FALSE` ở 30/30 runs không được xem là lỗi, bởi hai artifact sử dụng hai thứ tự trình bày khác nhau. Sau khi căn chỉnh theo feature name, toàn bộ 30 runs đều cho kết quả importance khớp.

Độ lệch lớn nhất của SHAP importance sau alignment nằm trong khoảng:

| Model | Range của `max_abs_diff` |
|---|---:|
| LightGBM | 9.71e-17 – 7.11e-15 |
| XGBoost | 3.73e-07 – 1.85e-06 |

Các sai lệch này tiếp tục nằm ở mức floating-point precision và không làm thay đổi feature importance được báo cáo.

## 4. Kiểm tra data và experimental protocol

Bên cạnh việc tính lại kết quả đầu ra, QA review còn tập trung vào các điểm có khả năng ảnh hưởng trực tiếp đến tính đúng đắn của experiment. Phạm vi kiểm tra bao gồm raw monthly input schema, monthly zone-hour aggregates, frozen top-50 zones, hourly feature table, lag alignment, cách xây dựng `median_lag_3w`, feature contract của A/B/C và temporal boundaries của HPO, Fold 1–4 và final test.

Ngoài ra, review kiểm tra việc tách riêng December final test khỏi các giai đoạn trước, expanding-window design, temporal leakage và tính nhất quán của semantic row keys trong SHAP sampling. Các kiểm tra này không phát hiện vấn đề làm thay đổi protocol đã được định nghĩa cho core experiment.

## 5. Kết luận

Kết quả QA review cho thấy các đầu ra của core experiment nhất quán với các artifact được dùng để kiểm tra lại. Cụ thể, 30/30 core runs khớp khi tính lại prediction metrics và 30/30 runs khớp khi tính lại SHAP importance. Đồng thời, SHAP sample size, feature contract và các thành phần chính của temporal protocol đều phù hợp với thiết kế nghiên cứu.

Trên cơ sở đó, các kết quả được sử dụng trong technical report có đủ bằng chứng kiểm tra độc lập ở mức data, protocol và result verification. Không phát hiện inconsistency nào làm thay đổi các kết luận chính của nghiên cứu.
