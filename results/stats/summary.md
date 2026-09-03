# Kết quả tổng hợp — prediction performance & SHAP stability

> **Quy ước báo cáo theo proposal mục 2.5/2.6:** số liệu chính là **mean ± standard deviation** qua fold1-4; `final_test` (tháng 12) báo cáo riêng, không gộp vào mean/std. Proposal ghi rõ *"Không dùng coefficient of variation hoặc significance test trong core scope"* — nên CV chỉ xuất hiện ở mục **Diagnostic** phía dưới, ngoài spec, và **không dùng để rút kết luận**. Proposal cũng cảnh báo không kết luận một feature "ổn định hơn" chỉ dựa vào std khi mean importance chênh lệch lớn.

## 1. Prediction performance (mean ± std qua fold1-4)

| variant | model | mae_mean | mae_std | mae_final_test | rmse_mean | rmse_std | rmse_final_test | wape_mean | wape_std | wape_final_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | lightgbm | 23.6089 | 2.8194 | 28.1346 | 40.8728 | 6.9957 | 45.3906 | 0.0941 | 0.0076 | 0.1075 |
| A | xgboost | 23.7962 | 2.8965 | 28.6404 | 41.4368 | 7.0592 | 46.7248 | 0.0949 | 0.0078 | 0.1094 |
| B | lightgbm | 23.3506 | 2.6304 | 27.1988 | 40.3321 | 6.7366 | 43.7896 | 0.0931 | 0.0068 | 0.1039 |
| B | xgboost | 23.5962 | 2.7313 | 27.5173 | 41.2616 | 7.0501 | 44.7481 | 0.0941 | 0.0072 | 0.1052 |
| C | lightgbm | 23.4446 | 2.6889 | 27.5253 | 40.4974 | 6.7722 | 44.5175 | 0.0935 | 0.0071 | 0.1052 |
| C | xgboost | 23.6836 | 2.8219 | 28.2175 | 41.4433 | 7.0142 | 46.2460 | 0.0944 | 0.0075 | 0.1078 |

- MAE trung bình thấp nhất: **variant B / lightgbm** (23.351 ± 2.630).
- **Cảnh báo về effect size:** std qua fold lớn hơn nhiều lần khoảng chênh lệch giữa các variant (xem bảng paired bên dưới), nên **không** đọc bảng này như bằng chứng variant nào tốt hơn hẳn.

### 1b. So sánh có kiểm soát theo từng cặp (fold × model)

Cùng fold, cùng model, cùng tập row — chỉ khác feature set. Mô tả thuần, không phải significance test.

**Mốc nhiễu đúng cho thiết kế này là `std_delta_mae`** (độ phân tán của chính các hiệu số), *không phải* `unpaired_mae_level_std_A`. Cột sau chủ yếu phản ánh tháng 8 khó hơn tháng 11, mà chênh lệch giữa các tháng đã bị triệt tiêu khi so cùng fold — dùng nó làm mốc sẽ hạ thấp hiệu ứng một cách sai lệch.

| variant | scope | n_pairs | n_better_than_A | mean_delta_mae | std_delta_mae | se_delta_mae | mean_over_std | mean_over_se | worst_delta_mae | best_delta_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B | pooled_2models | 8 | 8 | -0.2291 | 0.1866 | 0.0660 | 1.2279 | 3.4730 | -0.0795 | -0.5348 |
| B | xgboost | 4 | 4 | -0.2000 | 0.1823 | 0.0911 | 1.0972 | 2.1944 | -0.0795 | -0.4715 |
| B | lightgbm | 4 | 4 | -0.2582 | 0.2139 | 0.1070 | 1.2073 | 2.4146 | -0.0871 | -0.5348 |
| C | pooled_2models | 8 | 8 | -0.1385 | 0.1087 | 0.0384 | 1.2742 | 3.6040 | -0.0292 | -0.3407 |
| C | xgboost | 4 | 4 | -0.1126 | 0.0808 | 0.0404 | 1.3928 | 2.7856 | -0.0472 | -0.2266 |
| C | lightgbm | 4 | 4 | -0.1643 | 0.1387 | 0.0693 | 1.1849 | 2.3697 | -0.0292 | -0.3407 |

**std hay SE?** Phát biểu ở đây là về *trung bình* hiệu số, nên mẫu số đúng là **SE = std/√n**, không phải std. `mean_over_std` chỉ mô tả độ phân tán của từng cặp; `mean_over_se` mới là tỉ lệ tương ứng với claim về trung bình.

**Cảnh báo về tính độc lập:** 8 cặp gộp = 2 model × 4 fold **dùng chung tập row**, nên chúng không độc lập và √8 là lạc quan. Vì vậy bảng có thêm dòng tách riêng từng model (n=4) — đây là mức granularity trung thực hơn.

- Variant **B** (gộp 2 model): tốt hơn A ở **8/8** cặp, **cùng dấu ở mọi cặp**. Hiệu số trung bình **-0.229 MAE** (std 0.187, SE 0.066) → mean/SE ≈ **3.47**.
- Variant **B** (chỉ `xgboost`): tốt hơn A ở **4/4** cặp, **cùng dấu ở mọi cặp**. Hiệu số trung bình **-0.200 MAE** (std 0.182, SE 0.091) → mean/SE ≈ **2.19**.
- Variant **B** (chỉ `lightgbm`): tốt hơn A ở **4/4** cặp, **cùng dấu ở mọi cặp**. Hiệu số trung bình **-0.258 MAE** (std 0.214, SE 0.107) → mean/SE ≈ **2.41**.
- Variant **C** (gộp 2 model): tốt hơn A ở **8/8** cặp, **cùng dấu ở mọi cặp**. Hiệu số trung bình **-0.138 MAE** (std 0.109, SE 0.038) → mean/SE ≈ **3.60**.
- Variant **C** (chỉ `xgboost`): tốt hơn A ở **4/4** cặp, **cùng dấu ở mọi cặp**. Hiệu số trung bình **-0.113 MAE** (std 0.081, SE 0.040) → mean/SE ≈ **2.79**.
- Variant **C** (chỉ `lightgbm`): tốt hơn A ở **4/4** cặp, **cùng dấu ở mọi cặp**. Hiệu số trung bình **-0.164 MAE** (std 0.139, SE 0.069) → mean/SE ≈ **2.37**.

→ Hiệu ứng **nhỏ nhưng nhất quán về hướng**: cùng dấu ở mọi cặp, và vẫn giữ tỉ lệ mean/SE > 2 khi tách riêng từng model. Không phải bị chìm trong nhiễu.

## 2. SHAP feature-level importance (mean ± std qua fold1-4)

| variant | model | feature | mean_importance | std_importance | importance_final_test |
| --- | --- | --- | --- | --- | --- |
| A | lightgbm | lag_168 | 23.8337 | 0.5491 | 23.7510 |
| A | lightgbm | lag_336 | 20.4582 | 0.9599 | 17.7656 |
| A | lightgbm | lag_504 | 7.2728 | 0.7503 | 8.0374 |
| A | xgboost | lag_168 | 23.2060 | 1.0706 | 22.8687 |
| A | xgboost | lag_336 | 21.2809 | 1.2723 | 19.9465 |
| A | xgboost | lag_504 | 7.4622 | 0.4939 | 8.8782 |
| B | lightgbm | median_lag_3w | 52.8974 | 2.3995 | 51.8886 |
| B | xgboost | median_lag_3w | 55.7821 | 2.0505 | 53.9992 |
| C | lightgbm | lag_168 | 6.4372 | 0.4621 | 6.7270 |
| C | lightgbm | median_lag_3w | 52.3622 | 1.7720 | 48.8031 |
| C | xgboost | lag_168 | 6.7662 | 0.5608 | 7.2927 |
| C | xgboost | median_lag_3w | 47.6708 | 0.7872 | 47.2628 |

- Đọc std **cùng với** mean: các feature ở đây chênh nhau nhiều lần về mean importance (vd. `lag_504` ≈ 7 so với `lag_168` ≈ 23), nên **không** so std trực tiếp giữa chúng để kết luận feature nào ổn định hơn — đúng cảnh báo proposal mục 2.6.

### 2b. final_test lệch bao nhiêu so với phân bố fold1-4?

fold1-4 đều là tháng 8-11 — regime tương đối đồng dạng. Tháng 12 (lễ/nghỉ) là phép thử regime khác. Cột `z_final_test` = (final_test − mean) / std của chính feature đó.

| variant | model | feature | mean_importance | std_importance | importance_final_test | z_final_test |
| --- | --- | --- | --- | --- | --- | --- |
| A | lightgbm | lag_336 | 20.4582 | 0.9599 | 17.7656 | -2.8049 |
| C | lightgbm | median_lag_3w | 52.3622 | 1.7720 | 48.8031 | -2.0085 |
| A | xgboost | lag_336 | 21.2809 | 1.2723 | 19.9465 | -1.0488 |
| B | xgboost | median_lag_3w | 55.7821 | 2.0505 | 53.9992 | -0.8695 |
| C | xgboost | median_lag_3w | 47.6708 | 0.7872 | 47.2628 | -0.5182 |
| B | lightgbm | median_lag_3w | 52.8974 | 2.3995 | 51.8886 | -0.4204 |
| A | xgboost | lag_168 | 23.2060 | 1.0706 | 22.8687 | -0.3150 |
| A | lightgbm | lag_168 | 23.8337 | 0.5491 | 23.7510 | -0.1506 |
| C | lightgbm | lag_168 | 6.4372 | 0.4621 | 6.7270 | 0.6272 |
| C | xgboost | lag_168 | 6.7662 | 0.5608 | 7.2927 | 0.9389 |
| A | lightgbm | lag_504 | 7.2728 | 0.7503 | 8.0374 | 1.0192 |
| A | xgboost | lag_504 | 7.4622 | 0.4939 | 8.8782 | 2.8670 |

- **3/12** dòng lệch quá **2σ**, **5/12** dòng lệch quá 1σ so với phân bố fold1-4 của chính nó.
- Dịch chuyển **có hướng, nhất quán giữa các model**: `lag_336` **giảm ở mọi model** (z: -2.80, -1.05); `lag_504` **tăng ở mọi model** (z: +1.02, +2.87); `median_lag_3w` **giảm ở mọi model** (z: -2.01, -0.87, -0.52, -0.42).

#### Tách confound train-size

`final_test` vừa là tháng lệch regime, vừa là fold có **train set lớn nhất** (fold1=229,200, fold2=266,400, fold3=302,400, fold4=339,600, final_test=375,600 row) — hai cách giải thích bị lẫn vào nhau. Cách tách: nếu do train-size/drift thì xu hướng phải **đã có sẵn** và đơn điệu qua fold1-4, `final_test` chỉ nối dài; nếu do regime thì fold1-4 phẳng còn `final_test` **gãy ra khỏi** dải giá trị fold1-4.

| variant | model | feature | trend_rho_fold1_4 | z_final_test | final_inside_cv_range | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| A | lightgbm | lag_168 | 0.8000 | -0.1506 | True | trong dải fold1-4 (không phải dịch chuyển) |
| A | lightgbm | lag_336 | 0.0000 | -2.8049 | False | gãy khỏi xu hướng fold1-4 → regime là cách đọc hợp lý hơn |
| A | lightgbm | lag_504 | 0.4000 | 1.0192 | True | trong dải fold1-4 (không phải dịch chuyển) |
| A | xgboost | lag_168 | 0.8000 | -0.3150 | True | trong dải fold1-4 (không phải dịch chuyển) |
| A | xgboost | lag_336 | 0.0000 | -1.0488 | True | trong dải fold1-4 (không phải dịch chuyển) |
| A | xgboost | lag_504 | 0.8000 | 2.8670 | False | nối dài xu hướng sẵn có → LẪN confound train-size |
| B | lightgbm | median_lag_3w | 0.8000 | -0.4204 | True | trong dải fold1-4 (không phải dịch chuyển) |
| B | xgboost | median_lag_3w | 1.0000 | -0.8695 | True | trong dải fold1-4 (không phải dịch chuyển) |
| C | lightgbm | lag_168 | 1.0000 | 0.6272 | True | trong dải fold1-4 (không phải dịch chuyển) |
| C | lightgbm | median_lag_3w | 0.8000 | -2.0085 | False | gãy khỏi xu hướng fold1-4 → regime là cách đọc hợp lý hơn |
| C | xgboost | lag_168 | 1.0000 | 0.9389 | True | trong dải fold1-4 (không phải dịch chuyển) |
| C | xgboost | median_lag_3w | 0.4000 | -0.5182 | True | trong dải fold1-4 (không phải dịch chuyển) |

*Cảnh báo:* `trend_rho_fold1_4` tính trên **4 điểm**, chỉ nhận vài giá trị rời rạc (0, ±0.4, ±0.8, ±1.0) nên đây là kiểm tra thô, không phải bằng chứng chắc.

- **2** dòng gãy khỏi xu hướng fold1-4 (regime là cách đọc hợp lý hơn), **1** dòng chỉ nối dài xu hướng sẵn có (lẫn confound train-size), **9** dòng nằm gọn trong dải fold1-4 (không phải dịch chuyển thật).
- **Đọc lại kết luận, đã thu hẹp:** phần `lag_504` **tăng** không còn đứng vững như bằng chứng regime — nó đã tăng dần sẵn qua fold1-4 khi train set lớn dần, `final_test` chỉ nối tiếp. Phần trụ được là `lag_336` (và `median_lag_3w`): fold1-4 **không có xu hướng** rồi `final_test` rơi xuống dưới toàn bộ dải — đây mới là gãy thật.
- Vì vậy phát biểu đúng **không phải** "credit dịch từ lag 2 tuần sang lag 3 tuần" (nửa sau lẫn confound), mà là: **tháng 12 làm gãy quỹ đạo của `lag_336`/`median_lag_3w`**, trong khi bốn fold liền kề — vốn là các tháng đồng dạng và train set tăng đều — chưa bao giờ thử thách được điều đó.

## 3. Weekly-group importance (mean ± std qua fold1-4)

| variant | model | mean_group_importance | std_group_importance | group_importance_final_test |
| --- | --- | --- | --- | --- |
| A | lightgbm | 51.5646 | 1.4820 | 49.5540 |
| A | xgboost | 51.9490 | 2.0477 | 51.6935 |
| B | lightgbm | 52.8974 | 2.3995 | 51.8886 |
| B | xgboost | 55.7821 | 2.0505 | 53.9992 |
| C | lightgbm | 58.7994 | 2.2188 | 55.5301 |
| C | xgboost | 54.4370 | 1.2156 | 54.5555 |

- **Cảnh báo so sánh:** số feature trong nhóm khác nhau giữa các variant (A=3, B=1, C=2). Với variant B nhóm chỉ có 1 feature nên **group importance ≡ feature importance** — chênh lệch giữa các variant ở bảng này một phần là do **định nghĩa metric** (tổng trên số feature khác nhau), chưa thể quy hết cho hành vi model.

## 4. Rank stability (thước đo bổ sung — **không** nằm trong spec proposal)

*Ghi chú phạm vi:* proposal mục 2.6 chỉ quy định báo cáo mean ± std, và cấm CV. Rank stability **không nằm trong spec** — nó chỉ không bị cấm. Đây là thước đo bổ sung do nhóm thêm vào, không phải "thước đo đúng spec".

Spearman giữa từng cặp fold trên feature ranking. Rank không có mean ở mẫu số nên không dính artifact "mean nhỏ → CV to".

**Hai cột Spearman, phải đọc cột non-zone:** ranking đầy đủ có 58 feature, trong đó 50 là zone one-hot với importance rất nhỏ và thứ tự ổn định một cách tầm thường — chúng đẩy Spearman toàn cục lên ~0.98 bất kể weekly lag hành xử ra sao. Cột `non_zone` (chỉ 8 feature thật) mới có ý nghĩa.

**Giới hạn của thước đo:** rank **không nhạy với biên độ** — importance có thể dao động hàng chục phần trăm mà thứ hạng vẫn y nguyên (xem cột `importance_swing_pct`). Rank ổn định **không loại trừ** magnitude bất ổn.

| variant | model | feature | mean_pairwise_spearman_all_features | mean_pairwise_spearman_non_zone | ranks_by_fold | rank_range | importance_swing_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | lightgbm | lag_168 | 0.9835 | 1.0000 | 2→2→2→2 | 0 | 4.8967 |
| A | lightgbm | lag_336 | 0.9835 | 1.0000 | 3→3→3→3 | 0 | 11.3450 |
| A | lightgbm | lag_504 | 0.9835 | 1.0000 | 6→6→6→6 | 0 | 22.2059 |
| A | xgboost | lag_168 | 0.9752 | 1.0000 | 2→2→2→2 | 0 | 11.0025 |
| A | xgboost | lag_336 | 0.9752 | 1.0000 | 3→3→3→3 | 0 | 14.1811 |
| A | xgboost | lag_504 | 0.9752 | 1.0000 | 6→6→6→6 | 0 | 15.1318 |
| B | lightgbm | median_lag_3w | 0.9827 | 0.9714 | 1→1→2→1 | 1 | 9.9448 |
| B | xgboost | median_lag_3w | 0.9772 | 1.0000 | 1→1→1→1 | 0 | 8.5557 |
| C | lightgbm | lag_168 | 0.9811 | 1.0000 | 5→5→5→5 | 0 | 16.6592 |
| C | lightgbm | median_lag_3w | 0.9811 | 1.0000 | 1→1→1→1 | 0 | 7.8214 |
| C | xgboost | lag_168 | 0.9706 | 0.9821 | 5→5→5→5 | 0 | 19.8268 |
| C | xgboost | median_lag_3w | 0.9706 | 0.9821 | 1→2→2→2 | 1 | 3.1948 |

- Spearman toàn bộ feature: **A**=0.9794, **B**=0.9800, **C**=0.9759 (bị 50 zone dummy làm phồng, **không dùng để kết luận**).
- Spearman chỉ trên feature non-zone: **A**=1.0000, **B**=0.9857, **C**=0.9911 — đây mới là con số đáng đọc.
- Dao động thứ hạng lớn nhất của một weekly-lag feature qua fold1-4: **1** bậc — nhưng cùng lúc đó biên độ importance dao động tới **22.2%** (`lag_504`, variant A/lightgbm). Đây chính là minh hoạ rank ổn định nhưng magnitude thì không.
- **Đụng trần:** variant A/B/C đạt Spearman = **1.0000**, tức giá trị tối đa của thang đo. Khi một variant đã kịch trần thì thước đo **không còn khả năng phân biệt** — đây là lý do mạnh hơn cả việc rank không nhạy với biên độ.
- Phát biểu đúng: **thước đo rank không phân biệt được** A/B/C ở đây (đụng trần + không nhạy biên độ) — không đồng nghĩa với "không có khác biệt".

## 5. Diagnostic (ngoài spec proposal — không dùng để kết luận)

### 5a. CV = std/mean — vì sao không dùng để kết luận

| variant | model | feature | mean_importance | std_importance | cv |
| --- | --- | --- | --- | --- | --- |
| A | lightgbm | lag_504 | 7.2728 | 0.7503 | 0.1032 |
| C | xgboost | lag_168 | 6.7662 | 0.5608 | 0.0829 |
| C | lightgbm | lag_168 | 6.4372 | 0.4621 | 0.0718 |
| A | xgboost | lag_504 | 7.4622 | 0.4939 | 0.0662 |
| A | xgboost | lag_336 | 21.2809 | 1.2723 | 0.0598 |
| A | lightgbm | lag_336 | 20.4582 | 0.9599 | 0.0469 |
| A | xgboost | lag_168 | 23.2060 | 1.0706 | 0.0461 |
| B | lightgbm | median_lag_3w | 52.8974 | 2.3995 | 0.0454 |
| B | xgboost | median_lag_3w | 55.7821 | 2.0505 | 0.0368 |
| C | lightgbm | median_lag_3w | 52.3622 | 1.7720 | 0.0338 |
| A | lightgbm | lag_168 | 23.8337 | 0.5491 | 0.0230 |
| C | xgboost | median_lag_3w | 47.6708 | 0.7872 | 0.0165 |

- Tương quan Spearman giữa `mean_importance` và `cv` trên 12 dòng: **-0.839** — CV gần như bị quyết định bởi độ lớn của mean, đúng artifact mẫu số nhỏ mà proposal mục 2.6 cảnh báo.
- Cụ thể: 4 feature có CV cao nhất đều có mean ≤ **7.46**, trong khi 4 feature có CV thấp nhất đều có mean ≥ **23.83**. Nói "feature CV cao thì kém ổn định hơn" ở đây thực chất chỉ đang nói "feature đó có importance nhỏ hơn".

### 5b. Sensitivity: công thức weekly-group importance

Proposal mục 2.6 định nghĩa group importance = **Σ_j mean_i |φ_ij|** (tổng mean |φ| từng feature) — code đang implement đúng công thức này. Cách thay thế **mean_i |Σ_j φ_ij|** (mean của |tổng φ| trong từng sample) mới đúng nghĩa "đóng góp của cả nhóm", vì với feature tương quan các φ có thể triệt tiêu dấu nhau trong cùng 1 sample. `cancellation_ratio` = alt / proposal; càng nhỏ hơn 1 nghĩa là triệt tiêu dấu càng nhiều.

| variant | model | group_proposal_sum_mean_abs_mean | group_alt_mean_abs_sum_mean | group_alt_mean_abs_sum_std | cancellation_ratio_mean |
| --- | --- | --- | --- | --- | --- |
| A | lightgbm | 51.5646 | 49.8913 | 1.4306 | 0.9676 |
| A | xgboost | 51.9490 | 50.4062 | 1.8668 | 0.9704 |
| B | lightgbm | 52.8974 | 52.8974 | 2.3995 | 1.0000 |
| B | xgboost | 55.7821 | 55.7821 | 2.0505 | 1.0000 |
| C | lightgbm | 58.7994 | 58.0471 | 2.3071 | 0.9872 |
| C | xgboost | 54.4370 | 54.1960 | 1.1529 | 0.9956 |

- Mức triệt tiêu dấu mạnh nhất ở **variant A / lightgbm** (ratio=0.9676). Variant có nhóm 1 feature thì ratio = 1 theo định nghĩa (không có gì để triệt tiêu).

## 6. Đánh giá (proposal mục 6, câu hỏi 3)

**Variant B so với A:**

- `xgboost`: MAE giảm 0.8% (23.796 → 23.596); weekly-group importance 51.95 ± 2.05 → 55.78 ± 2.05.
- `lightgbm`: MAE giảm 1.1% (23.609 → 23.351); weekly-group importance 51.56 ± 1.48 → 52.90 ± 2.40.

**Variant C so với A:**

- `xgboost`: MAE giảm 0.5% (23.796 → 23.684); weekly-group importance 51.95 ± 2.05 → 54.44 ± 1.22.
- `lightgbm`: MAE giảm 0.7% (23.609 → 23.445); weekly-group importance 51.56 ± 1.48 → 58.80 ± 2.22.

**Kết luận:**

- Về **prediction**: gộp weekly-lag thắng baseline A ở **mọi** cặp fold × model. Hiệu số trung bình tối đa ~0.26 MAE; ngay cả khi tách riêng từng model (n=4, tránh giả định độc lập của 8 cặp gộp) thì mean/SE vẫn ≥ 2.19. Kết luận: hiệu ứng **nhỏ nhưng thật và nhất quán về hướng** — đủ để nói gộp feature không làm hại prediction, không đủ để nói nó cải thiện đáng kể.
- Về **explanation**: thước đo rank (mục 4, **bổ sung — không nằm trong spec**) **không phân biệt được** A/B/C: nó đụng trần (A = 1.0000) nên hết khả năng phân giải, và vốn cũng không nhạy với biên độ. Cơ chế "gộp feature tương quan làm credit dồn về một chỗ" là có thật về mặt lý thuyết với Tree SHAP `tree_path_dependent`, nhưng **không có thước đo nào hiện có đủ sức xác nhận hay bác bỏ nó trên dữ liệu này**.
- **Attribution KHÔNG ổn định khi đổi regime:** dù fold1-4 rất ổn định, tháng 12 (mục 2b) cho thấy credit dịch có hệ thống từ lag ngắn sang lag dài. Bốn fold liền kề là các tháng đồng dạng, nên chúng chưa từng thực sự thử thách tính ổn định mà đề tài muốn đo.
- **Không so sánh được ở group-level:** số feature trong nhóm khác nhau giữa các variant (mục 3), nên chênh lệch ở đó lẫn confound định nghĩa metric. Sign cancellation đã kiểm tra (mục 5b) và không phải nguyên nhân.
- **Chưa trả lời được:** biến động qua fold hiện chưa tách khỏi nhiễu nội tại của chính ước lượng SHAP (chỉ 1 seed, 1 tập 5.000 row mỗi fold). Thiếu baseline nhiễu theo seed/bootstrap thì mọi phát biểu về "ổn định" đều thiếu mốc so sánh — đây là hạn chế lớn nhất hiện tại.

## Chi tiết file

- `performance_summary.csv` — MAE/RMSE/WAPE raw theo từng fold.
- `performance_aggregated.csv` — MAE/RMSE/WAPE mean ± std theo (variant, model).
- `paired_mae_comparison.csv` — so sánh MAE theo từng cặp (fold × model) so với baseline A.
- `feature_importance_stability.csv` — SHAP importance mean/std theo từng weekly-lag feature (cột `cv` là diagnostic ngoài spec).
- `weekly_group_stability.csv` — weekly-group importance mean/std (cột `cv` là diagnostic ngoài spec).
- `rank_stability.csv` — Spearman giữa các fold + dao động thứ hạng weekly-lag feature.
- `group_formula_sensitivity.csv` — so sánh 2 công thức group importance (proposal vs mean|Σφ|).
- `plots/{variant}_feature_trend.png` — xu hướng importance từng feature qua fold.
- `plots/cv_comparison.png` — biểu đồ diagnostic CV (ngoài spec).