# Kết quả tổng hợp: prediction performance & SHAP stability

> **Phạm vi báo cáo theo protocol.** Số liệu là **mean ± sample standard deviation** qua 4 fold; `final_test` (tháng 12) báo cáo riêng, không gộp vào mean/std. Core không dùng coefficient of variation hoặc significance test và không mở rộng ngoài data, model, metric, split và SHAP settings đã khóa. Vì vậy báo cáo này **chỉ** gồm các chỉ số trên, không kèm CV, rank stability, paired standard error hay z-score.

> **Cách đọc standard deviation:** std được diễn giải **cùng với** mean importance. Không kết luận một feature "ổn định hơn" chỉ dựa vào std khi mức mean importance khác nhau quá lớn.

## 1. Tương quan giữa ba weekly lag

Pearson correlation tính theo từng zone trên 50 frozen zone. Protocol không đặt trước ngưỡng để gọi là "cao"; giá trị quan sát được quyết định mức độ mạnh của kết luận.

| cặp lag | mean r | std r | n zone |
| --- | --- | --- | --- |
| lag_168_vs_lag_336 | 0.9077 | 0.0241 | 50 |
| lag_168_vs_lag_504 | 0.8942 | 0.0238 | 50 |
| lag_336_vs_lag_504 | 0.8956 | 0.0245 | 50 |

- Cả ba cặp nằm trong khoảng **0.8942** tới **0.9077**, std lớn nhất chỉ **0.0245** trên 50 zone. Tương quan **cao và đồng nhất giữa các zone**, premise của đề tài đứng vững.

## 2. Prediction performance (mean ± std qua 4 fold)

MAE là metric chính; RMSE và WAPE là metric bổ sung. WAPE tính theo phần trăm.

| variant | model | mae_mean | mae_std | mae_final_test | rmse_mean | rmse_std | rmse_final_test | wape_mean | wape_std | wape_final_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | lightgbm | 23.6089 | 2.8194 | 28.1346 | 40.8728 | 6.9957 | 45.3906 | 9.4126 | 0.7606 | 10.7512 |
| A | xgboost | 23.7007 | 2.8671 | 28.2399 | 41.4736 | 7.1204 | 46.1812 | 9.4483 | 0.7685 | 10.7914 |
| B | lightgbm | 23.3506 | 2.6304 | 27.1988 | 40.3321 | 6.7366 | 43.7896 | 9.3109 | 0.6797 | 10.3936 |
| B | xgboost | 23.5303 | 2.7266 | 27.2513 | 41.4022 | 7.2300 | 44.4946 | 9.3819 | 0.7162 | 10.4136 |
| C | lightgbm | 23.4446 | 2.6889 | 27.5253 | 40.4974 | 6.7722 | 44.5175 | 9.3482 | 0.7078 | 10.5183 |
| C | xgboost | 23.5952 | 2.7490 | 27.7334 | 41.4869 | 7.0366 | 45.7414 | 9.4075 | 0.7230 | 10.5979 |

- MAE trung bình thấp nhất: **variant B / lightgbm** (23.351 ± 2.630).
- Khoảng chênh lệch mean MAE giữa 6 tổ hợp là **0.350**, trong khi std qua fold lên tới **2.867**. Std ở đây phản ánh mức khó khác nhau giữa các tháng đánh giá, nên bảng này mô tả kết quả chứ không phải bằng chứng variant nào tốt hơn hẳn.

## 3. SHAP feature-level importance (mean ± std qua 4 fold)

Protocol định nghĩa `I_{j,f}` = mean |phi| của feature j trên 5.000 sample row của fold f. Bảng dưới là mean và sample standard deviation của `I_{j,f}` qua các fold, kèm cột final_test tách riêng.

| variant | model | feature | mean_importance | std_importance | importance_final_test |
| --- | --- | --- | --- | --- | --- |
| A | lightgbm | lag_168 | 24.1246 | 0.5608 | 23.8256 |
| A | lightgbm | lag_336 | 20.7284 | 0.9223 | 18.0264 |
| A | lightgbm | lag_504 | 7.3669 | 0.7729 | 8.2421 |
| A | xgboost | lag_168 | 22.0284 | 1.5563 | 22.5355 |
| A | xgboost | lag_336 | 21.2158 | 1.3553 | 18.3263 |
| A | xgboost | lag_504 | 7.4845 | 0.4122 | 8.7029 |
| B | lightgbm | median_lag_3w | 53.6957 | 2.4655 | 52.5064 |
| B | xgboost | median_lag_3w | 54.1682 | 1.8204 | 51.0711 |
| C | lightgbm | lag_168 | 6.4738 | 0.4916 | 6.6010 |
| C | lightgbm | median_lag_3w | 53.1915 | 1.8343 | 49.4400 |
| C | xgboost | lag_168 | 4.8993 | 0.5145 | 5.2161 |
| C | xgboost | median_lag_3w | 54.7119 | 1.9939 | 52.0773 |

- Mean importance giữa các feature chênh nhau rất lớn (từ **4.90** ở `lag_168` [C/xgboost] tới **54.71** ở `median_lag_3w` [C/xgboost]), nên **không** so std trực tiếp giữa chúng để xếp hạng độ ổn định, đúng quy ước đọc kết quả của project.

## 4. Weekly-group importance (mean ± std qua 4 fold)

Protocol định nghĩa `I_weekly,f` là tổng `I_{j,f}` của các weekly feature trong variant: **A** = `lag_168` + `lag_336` + `lag_504`; **B** = `median_lag_3w`; **C** = `lag_168` + `median_lag_3w`.

| variant | model | mean_group_importance | std_group_importance | group_importance_final_test |
| --- | --- | --- | --- | --- |
| A | lightgbm | 52.2199 | 1.5302 | 50.0941 |
| A | xgboost | 50.7287 | 2.0070 | 49.5646 |
| B | lightgbm | 53.6957 | 2.4655 | 52.5064 |
| B | xgboost | 54.1682 | 1.8204 | 51.0711 |
| C | lightgbm | 59.6653 | 2.3123 | 56.0410 |
| C | xgboost | 59.6112 | 2.4962 | 57.2934 |

- **Cảnh báo so sánh:** số feature trong nhóm khác nhau giữa các variant (A=3, B=1, C=2). Với variant B nhóm chỉ có 1 feature nên **group importance ≡ feature importance**. Chênh lệch giữa các variant ở bảng này một phần đến từ **định nghĩa metric** (tổng trên số feature khác nhau), chưa thể quy hết cho hành vi model.

## 5. Trả lời câu hỏi nghiên cứu

**Câu hỏi 1: ba weekly lag tương quan tới đâu và có đồng nhất giữa các zone không?** mean r **0.89** tới **0.91**, std r tối đa **0.024** trên 50 zone (chi tiết ở mục 1).

**Câu hỏi 2: mean/std của SHAP importance qua temporal fold thay đổi thế nào ở Variant A/B/C?**

- Variant **A** / `xgboost`: `lag_168` 22.03 ± 1.56; `lag_336` 21.22 ± 1.36; `lag_504` 7.48 ± 0.41.
- Variant **A** / `lightgbm`: `lag_168` 24.12 ± 0.56; `lag_336` 20.73 ± 0.92; `lag_504` 7.37 ± 0.77.
- Variant **B** / `xgboost`: `median_lag_3w` 54.17 ± 1.82.
- Variant **B** / `lightgbm`: `median_lag_3w` 53.70 ± 2.47.
- Variant **C** / `xgboost`: `median_lag_3w` 54.71 ± 1.99; `lag_168` 4.90 ± 0.51.
- Variant **C** / `lightgbm`: `median_lag_3w` 53.19 ± 1.83; `lag_168` 6.47 ± 0.49.

**Câu hỏi 3: B/C ảnh hưởng thế nào đến MAE, RMSE, WAPE và weekly-group SHAP stability so với A, và xu hướng có nhất quán giữa LightGBM và XGBoost không?**

*Variant B so với A:*

- `xgboost`: MAE giảm 0.7% (23.701 → 23.530); RMSE giảm 0.2% (41.474 → 41.402); WAPE giảm 0.7% (9.448 → 9.382). Weekly-group importance 50.73 ± 2.01 → 54.17 ± 1.82.
- `lightgbm`: MAE giảm 1.1% (23.609 → 23.351); RMSE giảm 1.3% (40.873 → 40.332); WAPE giảm 1.1% (9.413 → 9.311). Weekly-group importance 52.22 ± 1.53 → 53.70 ± 2.47.

*Variant C so với A:*

- `xgboost`: MAE giảm 0.4% (23.701 → 23.595); RMSE gần như không đổi (41.474 → 41.487); WAPE giảm 0.4% (9.448 → 9.408). Weekly-group importance 50.73 ± 2.01 → 59.61 ± 2.50.
- `lightgbm`: MAE giảm 0.7% (23.609 → 23.445); RMSE giảm 0.9% (40.873 → 40.497); WAPE giảm 0.7% (9.413 → 9.348). Weekly-group importance 52.22 ± 1.53 → 59.67 ± 2.31.

- **Tính nhất quán giữa hai model (variant B):** mean MAE giảm ở **cả LightGBM và XGBoost** (xgboost -0.7%, lightgbm -1.1%).
- **Tính nhất quán giữa hai model (variant C):** mean MAE giảm ở **cả LightGBM và XGBoost** (xgboost -0.4%, lightgbm -0.7%).

- **Mức độ mạnh của kết luận:** chênh lệch mean MAE giữa các variant nhỏ hơn nhiều so với std qua fold (tối đa 2.867). Core không dùng significance test, nên phát biểu dừng ở mức **mô tả**: gộp weekly lag không làm hại prediction, và hướng thay đổi nhất quán giữa hai model.
- **Về explanation stability:** so sánh weekly-group std giữa các variant bị lẫn confound định nghĩa metric (mục 3), còn so sánh feature-level std giữa các feature có mean chênh lệch lớn thì không được dùng std đơn độc để xếp hạng stability. Trong phạm vi core scope, số liệu mean ± std ở mục 2 và 3 là kết quả báo cáo được; kết luận mạnh hơn cần thước đo nằm ngoài spec hiện tại.

## Chi tiết file

- `performance_summary.csv`: MAE/RMSE/WAPE theo từng fold.
- `performance_aggregated.csv`: MAE/RMSE/WAPE mean ± std qua Fold 1-4 + cột final_test.
- `shap_importance_per_fold.csv`: `I_{j,f}` của weekly feature theo từng fold.
- `feature_importance_stability.csv`: mean ± std của `I_{j,f}` qua Fold 1-4 + cột final_test.
- `weekly_group_per_fold.csv`: weekly-group importance theo từng fold.
- `weekly_group_stability.csv`: weekly-group importance mean ± std + cột final_test.
- `hpo_comparison.csv`: baseline vs tuned trên HPO split.
- `plots/{variant}_feature_trend.png`: `I_{j,f}` của từng weekly feature qua các fold.

Correlation ở mục 1 do tầng Data sinh ra, module này chỉ đọc lại: `data/processed/correlation_summary.csv` và `correlation_by_zone.csv`.