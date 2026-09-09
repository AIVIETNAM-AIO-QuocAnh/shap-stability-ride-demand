# shap-stability-ride-demand

Dự án AIO Conquer Module 03: Độ ổn định của SHAP dưới các feature time-series tương quan cho
forecasting ride-demand theo zone và hour.

Câu hỏi nghiên cứu chính (`m03-proposal.pdf` mục 1): 3 weekly lag (`lag_168`, `lag_336`, `lag_504`)
của demand theo giờ NYC tương quan ra sao; SHAP importance của chúng ổn định thế nào qua các fold
theo thời gian; và việc gộp các feature này (Variant B/C) ảnh hưởng thế nào đến explanation
stability và forecasting performance so với giữ riêng lẻ (Variant A)?

## Phạm vi báo cáo

Báo cáo này chỉ dùng các chỉ số được proposal quy định:

| Nguồn | Chỉ số |
|---|---|
| Mục 2.2 | Pearson correlation của 3 cặp weekly lag theo từng zone; mean và std trên 50 zone |
| Mục 2.5 | MAE, RMSE, WAPE theo từng fold; **mean ± sample standard deviation** qua Fold 1–4; `final_test` (tháng 12) báo cáo **riêng** |
| Mục 2.6 | `I_{j,f}` = mean \|φ\| của feature *j* trên 5.000 sample row của fold *f*; mean ± sample std qua Fold 1–4 |
| Mục 2.6 | `I_weekly,f` = tổng `I_{j,f}` của các weekly feature trong variant; mean ± sample std qua Fold 1–4 |
| Mục 3.2 | Baseline vs tuned trên HPO split |

Proposal mục 2.6 ghi rõ *"Không dùng coefficient of variation hoặc significance test trong core
scope"*, và mục 6 ghi *"Không cần thêm model, dataset, statistical test, deployment hoặc experiment
ngoài phạm vi trên để coi core project là hoàn thành"*. Vì vậy báo cáo **không** dùng coefficient of
variation, rank/Spearman stability, paired standard error, z-score hay significance test. Các thước
đo đó nếu cần thì thuộc phần follow-up, liệt kê ở cuối file.

Proposal mục 2.6 cũng cảnh báo: std phải được **đọc cùng mean importance**, và không kết luận một
feature "ổn định hơn" chỉ dựa vào std khi mức mean importance khác nhau quá lớn. Ràng buộc này chi
phối mọi kết luận ở mục 2 và 3 dưới đây.

Số liệu đầy đủ: [results/stats/summary.md](results/stats/summary.md). Pipeline:
[src/pipeline/README.md](src/pipeline/README.md). Tổng hợp:
[src/analysis/README.md](src/analysis/README.md).

## 1. Tương quan giữa 3 weekly lag (proposal mục 6, câu hỏi 1)

`data/processed/correlation_summary.csv`, Pearson theo từng zone trong 50 zone, giai đoạn
22/01–31/07/2025:

| cặp lag | mean r | std r | n zone |
|---|---|---|---|
| `lag_168` × `lag_336` | 0.9077 | 0.0241 | 50/50 |
| `lag_168` × `lag_504` | 0.8942 | 0.0238 | 50/50 |
| `lag_336` × `lag_504` | 0.8956 | 0.0245 | 50/50 |

Tương quan **cao và đồng nhất giữa các zone**: cả 3 cặp đều quanh 0.89–0.91, std chỉ ~0.024 trên 50
zone, không zone nào undefined. Proposal mục 2.2 chủ ý không đặt trước ngưỡng để gọi là "cao"; với
mức 0.89–0.91 và độ phân tán nhỏ như vậy, premise của đề tài đứng vững: ba weekly lag mang
thông tin chồng lấn mạnh.

## 2. Mean ± std của SHAP importance qua Fold 1–4 (proposal mục 6, câu hỏi 2)

`results/stats/feature_importance_stability.csv`. `final_test` là cột riêng, không tham gia mean/std.

| variant | model | feature | mean ± std | final_test |
|---|---|---|---|---|
| A | lightgbm | `lag_168` | 24.12 ± 0.56 | 23.83 |
| A | lightgbm | `lag_336` | 20.73 ± 0.92 | 18.03 |
| A | lightgbm | `lag_504` | 7.37 ± 0.77 | 8.24 |
| A | xgboost | `lag_168` | 22.03 ± 1.56 | 22.54 |
| A | xgboost | `lag_336` | 21.22 ± 1.36 | 18.33 |
| A | xgboost | `lag_504` | 7.48 ± 0.41 | 8.70 |
| B | lightgbm | `median_lag_3w` | 53.70 ± 2.47 | 52.51 |
| B | xgboost | `median_lag_3w` | 54.17 ± 1.82 | 51.07 |
| C | lightgbm | `lag_168` | 6.47 ± 0.49 | 6.60 |
| C | lightgbm | `median_lag_3w` | 53.19 ± 1.83 | 49.44 |
| C | xgboost | `lag_168` | 4.90 ± 0.51 | 5.22 |
| C | xgboost | `median_lag_3w` | 54.71 ± 1.99 | 52.08 |

Xu hướng qua từng fold:

| Variant A | Variant B | Variant C |
|---|---|---|
| ![Variant A feature trend](results/stats/plots/A_feature_trend.png) | ![Variant B feature trend](results/stats/plots/B_feature_trend.png) | ![Variant C feature trend](results/stats/plots/C_feature_trend.png) |

**Đọc được gì từ mean:**

- **Variant A** phân bổ credit theo đúng khoảng cách thời gian: `lag_168` (1 tuần) ≈ 22–24,
  `lag_336` (2 tuần) ≈ 21, `lag_504` (3 tuần) chỉ ≈ 7.4. Thứ tự này giống nhau ở cả hai model.
- **Variant C là quan sát đáng chú ý nhất.** Cùng là `lag_168`, nhưng khi đứng cạnh
  `median_lag_3w` thì importance của nó sụp từ ≈ 22–24 (ở A) xuống còn **4.90–6.47**, tức mất khoảng
  3/4 credit, nhất quán ở cả hai model. Khi có mặt một đại diện gộp tương quan cao với nó, gần như
  toàn bộ credit dồn về đại diện gộp.
- **Variant B** gom toàn bộ nhóm vào 1 feature, mean ≈ 53.7–54.2.

**Không đọc được gì từ std, và đây là kết luận chứ không phải thiếu sót.** Proposal mục 2.6 cấm
so std giữa các feature có mean chênh lệch lớn. Ở đây mean trải từ **4.90** (`lag_168` ở
C/xgboost) tới **54.71** (`median_lag_3w` ở C/xgboost), chênh hơn 11 lần. Vì vậy **không thể** dùng bảng này để nói
feature nào hay variant nào "ổn định hơn" ở mức feature-level. Câu hỏi 2 của proposal được trả lời
đúng ở dạng nó cho phép: **bảng mean ± std ở trên là kết quả**, và mức mean chênh lệch quá lớn khiến
so sánh độ ổn định giữa các variant ở tầng feature không thực hiện được trong core scope.

## 3. Weekly-group importance (proposal mục 2.6)

`results/stats/weekly_group_stability.csv`. `I_weekly,f` = tổng `I_{j,f}` của các weekly feature
trong variant: A = `lag_168`+`lag_336`+`lag_504`; B = `median_lag_3w`; C = `lag_168`+`median_lag_3w`.

| variant | model | mean ± std | final_test |
|---|---|---|---|
| A | lightgbm | 52.22 ± 1.53 | 50.09 |
| A | xgboost | 50.73 ± 2.01 | 49.56 |
| B | lightgbm | 53.70 ± 2.47 | 52.51 |
| B | xgboost | 54.17 ± 1.82 | 51.07 |
| C | lightgbm | 59.67 ± 2.31 | 56.04 |
| C | xgboost | 59.61 ± 2.50 | 57.29 |

Ở tầng group, mean importance của ba variant gần nhau hơn nhiều so với tầng feature (50.7–59.7, chênh
~18%), nên so std ở đây ít vi phạm cảnh báo mục 2.6 hơn. Đọc thẳng:

- **Gộp feature không làm group importance ổn định hơn.** Variant A có std thấp nhất ở lightgbm
  (1.53), và C có std cao hơn A ở **cả hai model** (2.31 vs 1.53; 2.50 vs 2.01).
- Với **B thì không nhất quán giữa hai model**: std cao hơn A ở lightgbm (2.47 vs 1.53) nhưng thấp
  hơn A ở xgboost (1.82 vs 2.01). Proposal mục 6 câu 3 yêu cầu kiểm tra tính nhất quán giữa
  LightGBM và XGBoost. Ở tiêu chí này B **không** đạt.

Nói cách khác, giả thuyết ngầm của đề tài (gộp feature tương quan sẽ làm explanation ổn định hơn)
**không được số liệu ủng hộ** ở tầng group: hướng quan sát được là ngược lại với C, và không nhất
quán với B.

**Cảnh báo bắt buộc khi đọc bảng này:** số feature trong nhóm khác nhau giữa các variant (A=3, B=1,
C=2). Riêng variant B nhóm chỉ có 1 feature nên group importance **≡** feature importance. Chênh lệch
group-level giữa A/B/C vì vậy lẫn confound **định nghĩa metric** (tổng trên số feature khác nhau),
chưa thể quy hết cho hành vi model.

## 4. Ảnh hưởng của việc gộp feature lên prediction (proposal mục 6, câu hỏi 3)

`results/stats/performance_aggregated.csv`. MAE là metric chính; RMSE và WAPE bổ sung. WAPE là phần
trăm.

| variant | model | MAE mean ± std | MAE final_test | RMSE mean ± std | WAPE mean ± std |
|---|---|---|---|---|---|
| A | lightgbm | 23.609 ± 2.819 | 28.135 | 40.873 ± 6.996 | 9.413 ± 0.761 |
| A | xgboost | 23.701 ± 2.867 | 28.240 | 41.474 ± 7.120 | 9.448 ± 0.769 |
| B | lightgbm | **23.351 ± 2.630** | 27.199 | 40.332 ± 6.737 | 9.311 ± 0.680 |
| B | xgboost | 23.530 ± 2.727 | 27.251 | 41.402 ± 7.230 | 9.382 ± 0.716 |
| C | lightgbm | 23.445 ± 2.689 | 27.525 | 40.497 ± 6.772 | 9.348 ± 0.708 |
| C | xgboost | 23.595 ± 2.749 | 27.733 | 41.487 ± 7.037 | 9.408 ± 0.723 |

Chênh lệch so với baseline A:

| variant | model | MAE | RMSE | WAPE |
|---|---|---|---|---|
| B | lightgbm | −1.1% | −1.3% | −1.1% |
| B | xgboost | −0.7% | −0.2% | −0.7% |
| C | lightgbm | −0.7% | −0.9% | −0.7% |
| C | xgboost | −0.4% | ≈0% | −0.4% |

**Tính nhất quán giữa hai model** (điều proposal mục 6 câu 3 hỏi trực tiếp): mean MAE **giảm ở cả
LightGBM và XGBoost** với cả hai variant gộp. Hướng của RMSE và WAPE cũng vậy, trừ RMSE của C/xgboost
gần như không đổi (41.474 → 41.487).

**Mức độ mạnh của kết luận.** Khoảng chênh lệch mean MAE giữa toàn bộ 6 tổ hợp chỉ là **0.350**,
trong khi std qua fold lên tới **2.867**. Std đó chủ yếu phản ánh việc các tháng đánh giá khó dễ
khác nhau (MAE trung bình theo fold: tháng 8 ≈ 20.3, tháng 9 ≈ 23.5, tháng 10 ≈ 23.3, tháng 11 ≈
27.0), chứ không phải nhiễu của phép so sánh.
Proposal không cho phép dùng significance test trong core scope để tách hai nguồn biến thiên này, nên
phát biểu dừng ở mức mô tả:

> Gộp weekly lag **không làm hại** prediction: mean MAE và WAPE thấp hơn baseline A ở cả 4 tổ hợp
> variant × model, RMSE thấp hơn ở 3/4 và gần như không đổi ở tổ hợp còn lại (C/xgboost, +0.03%).
> Hướng thay đổi nhất quán ở cả hai model. Nhưng biên độ cải thiện (≤1.3%) nhỏ hơn nhiều so với biến
> thiên giữa các tháng, nên **không** kết luận được rằng B hoặc C thực sự tốt hơn A.

### Tác động của tuning (proposal mục 3.2)

`results/stats/hpo_comparison.csv`, trên HPO split (tháng 07/2025), chỉ Variant A:

| model | MAE baseline | MAE tuned | thay đổi |
|---|---|---|---|
| xgboost | 25.342 | 23.901 | −5.7% |
| lightgbm | 25.787 | 23.647 | −8.3% |

Best config của mỗi model được freeze sau bước này và dùng nguyên vẹn cho A/B/C ở Fold 1–4 và
final test, không tune lại theo variant hoặc fold.

## Thảo luận

Trong đúng phạm vi proposal cho phép, ba câu hỏi hoàn thành được trả lời như sau:

1. **Tương quan:** cao (0.89–0.91) và đồng nhất giữa 50 zone (std ≈ 0.024). Premise vững.
2. **Mean/std của SHAP importance qua fold ở A/B/C:** bảng mục 2 là kết quả. Phát hiện rõ nhất nằm ở
   **mean**, không ở std: khi thêm `median_lag_3w` vào cạnh `lag_168` (variant C), credit của
   `lag_168` sụp còn khoảng 1/4, nhất quán ở cả hai model. Còn so sánh độ ổn định ở tầng feature thì
   **không thực hiện được**: mean chênh nhau hơn 11 lần, đúng tình huống proposal mục 2.6 cấm so std.
3. **Ảnh hưởng của B/C:** prediction cải thiện nhẹ (≤1.3% MAE) và **nhất quán giữa hai model**, nhưng
   biên độ nhỏ hơn nhiều so với biến thiên giữa các tháng. Ở group-level, gộp feature **không** làm
   importance ổn định hơn: C có std cao hơn A ở cả hai model, còn B không nhất quán giữa hai model.

Kết quả tổng thể vì vậy là **bất đối xứng**: gộp feature tương quan **có** tác động rõ và nhất quán
lên *cách credit được phân bổ* (mục 2) nhưng **không** cho thấy lợi ích về *độ ổn định của credit*
(mục 3), trong khi lợi ích về prediction thì tồn tại nhưng quá nhỏ để khẳng định trong core scope.

### Hạn chế đã biết

- **Chưa có baseline nhiễu.** Biến động qua 4 fold chưa tách được khỏi nhiễu nội tại của chính ước
  lượng SHAP, vì mỗi fold chỉ dùng 1 seed và 1 tập 5.000 row. Không có mốc nhiễu này thì mọi
  phát biểu về "ổn định" đều thiếu điểm so sánh. Đây là hạn chế lớn nhất.
- **4 fold là mẫu rất nhỏ để ước lượng std**, và cả 4 đều là tháng 8–11, cùng một regime tương đối
  đồng dạng. Những gì đo được là độ ổn định *trong* một regime, chưa phải độ ổn định nói chung.
- **Train size đồng biến với thời gian.** Cửa sổ train mở rộng đều mỗi fold (229k → 266k → 302k →
  340k → 376k row), nên "fold muộn hơn" luôn đi kèm "train lớn hơn". Đây là confound cấu trúc của
  protocol split, không tách được bằng các chỉ số trong core scope.
- **Group-level lẫn confound định nghĩa metric**: số feature trong nhóm khác nhau giữa các variant
  (A=3, B=1, C=2), nên chênh lệch group-level không quy hết cho hành vi model được.
- **Không có ground truth** để phân biệt "ổn định" với "đúng". Một variant ổn định hơn hoàn toàn có
  thể chỉ vì nó ném bớt thông tin.
- **Artifact hiện tại được sinh trên môi trường khác `environment.yaml`.** Spec pin Python 3.10.9 /
  xgboost 2.1.4 / shap 0.47.0 / pandas 2.2.3; lần chạy này dùng Python 3.12.3 / xgboost 3.3.0 /
  shap 0.52.0 / pandas 3.0.2 (ghi trong `run_config.json` của từng run). Kết quả LightGBM tái lập
  chính xác giữa hai môi trường; kết quả XGBoost thì không. Muốn tái lập bit-for-bit toàn bộ thì
  phải dựng đúng environment trong `environment.yaml`.

### Ngoài core scope

Proposal mục 6 ghi *"Không cần thêm model, dataset, statistical test, deployment hoặc experiment
ngoài phạm vi trên"*, nên các hướng dưới đây **không** thuộc kết quả bàn giao; xếp theo mức độ nâng
chất lượng đề tài:

1. **Thêm trục nhiễu, không chỉ trục thời gian.** Chạy lại cùng một fold với nhiều seed cho tập 5.000
   row (và/hoặc bootstrap) để có baseline nhiễu. Nếu biến động qua fold không lớn hơn biến động qua
   seed thì mọi kết luận stability sụp, và biết được điều đó cũng đã là kết quả đáng viết. Đây là
   hướng đánh trúng hạn chế lớn nhất hiện tại.
2. **Thiết kế thước đo stability phù hợp.** Trong core scope chỉ có mean ± std, mà std lại không so
   được giữa các feature có mean chênh lệch lớn (mục 2). Cần thước đo nhạy với biên độ nhưng không
   chuẩn hoá bằng chính mean, ví dụ std của importance đã chuẩn hoá theo *tổng* importance toàn
   model. Lưu ý coefficient of variation **không** phải lời giải: nó chuẩn hoá đúng bằng đại lượng
   gây vấn đề, và proposal mục 2.6 đã cấm.
3. **Control bán tổng hợp để tách stability khỏi correctness.** Sinh target từ hàm đã biết trọng số
   trên 3 lag, rồi xem SHAP có phục hồi đúng attribution ở A/B/C không.
4. **Chọn fold theo regime, không chỉ theo thời gian liền kề**, và cố ý phá thế đồng biến
   train-size ↔ thời gian, để thực sự stress-test attribution.
5. **So `tree_path_dependent` với `interventional`**: gần như một dòng code, đánh trúng cơ chế
   correlation bóp méo attribution.
6. Các thước đo bổ sung như rank/Spearman stability, paired comparison, z-score của `final_test`: nếu
   dùng thì phải ghi rõ là ngoài spec, và không đưa vào bảng bàn giao core.

## Cách chạy lại

Từ project root, trong environment `shap-stability-ride-demand` (xem `environment.yaml`):

```bash
# 1. Data (cần data/raw/ với 12 file Parquet HVFHV 2025)
python -m src.data.download_raw
python -m src.data.freeze_zones
python -m src.data.build_panel
python -m src.data.correlation_analysis
python -m src.data.split_folds
python -m src.data.qa_checks

# 2. Pipeline: sample keys, HPO, 30 core run + SHAP, QA matrix, canonical summary
python run.py all

# 3. Analysis: bảng bàn giao + hình
python -m src.analysis.run_stats

# 4. Test contract của Pipeline (5 test)
python -m unittest src.test.test_pipeline
```

Protocol và đường dẫn dùng chung nằm trong `configs/data.yaml` (dữ liệu, split, variant) và
`configs/model.yaml` (model, HPO, SHAP). Không sửa code để đổi setting, sửa hai file này.

Artifact writer **không ghi đè** file đã tồn tại: muốn chạy lại sạch thì phải xoá `results/` và
`data/folds/*/shap_sample_keys.csv` trước. Chi tiết từng stage nằm trong
[src/pipeline/README.md](src/pipeline/README.md) và [src/data/README.md](src/data/README.md).

### Dashboard (ngoài core scope)

Công cụ trình bày, đọc lại artifact đã có, không train lại gì. Không thuộc kết quả bàn giao của
proposal. Xem [src/dashboard/README.md](src/dashboard/README.md). Gồm hai trang:

- **Forecast**: bản đồ demand theo zone và giờ, chọn được ngày, giờ, district, zone, variant,
  model; có badge riêng cho từng ngày lễ liên bang.
- **Experiment**: trình bày lại đúng các bảng của báo cáo này (mục 1 tới 4 ở trên cộng HPO),
  đọc thẳng từ `results/stats/` nên số luôn khớp.

Dependency của dashboard được khai báo trong `environment.yaml`. Với checkout mới, tạo environment
chung bằng `conda env create -f environment.yaml`; với environment hiện có, cập nhật bằng
`conda env update -f environment.yaml`. Sau đó activate environment và chạy dashboard:

```bash
conda activate shap-stability-ride-demand
python -m src.dashboard.build_zone_geojson
streamlit run dashboard_app.py
```

## Cấu trúc source

| Đường dẫn | Vai trò | Tài liệu |
|---|---|---|
| `configs/` | `data.yaml` và `model.yaml`, khoá toàn bộ protocol | (mô tả trong hai readme dưới) |
| `src/data/` | tải, aggregate, dựng panel, feature, correlation, split fold, QA | [src/data/README.md](src/data/README.md) |
| `src/pipeline/` | HPO, training, SHAP, artifact writer, QA matrix, canonical summary | [src/pipeline/README.md](src/pipeline/README.md) |
| `src/analysis/` | tổng hợp `results/` thành bảng bàn giao và `summary.md` | [src/analysis/README.md](src/analysis/README.md) |
| `src/test/` | 5 test contract của Pipeline | chạy bằng `python -m unittest src.test.test_pipeline` |
| `src/heatmap/` | shapefile và lookup NYC taxi zone, dữ liệu tĩnh làm nền bản đồ | [src/heatmap/README.md](src/heatmap/README.md) |
| `src/dashboard/` | dashboard Streamlit, ngoài core scope | [src/dashboard/README.md](src/dashboard/README.md) |
| `run.py` | entry point của Pipeline | [src/pipeline/README.md](src/pipeline/README.md) |
| `dashboard_app.py` | entry point của dashboard | [src/dashboard/README.md](src/dashboard/README.md) |

`m03-proposal.pdf` là spec gốc; `m03-workplan.pdf` là kế hoạch 14 ngày.

## Cấu trúc kết quả

```text
results/
├── run_matrix.csv                       # QA: 36 unit (2 HPO mode × 2 model + 30 core run)
├── hpo/{model}/                         # best_params.json, trials.csv, run_config.json
├── A/hpo/{model,model_baseline}/        # baseline vs tuned trên HPO split
├── {A,B,C}/{fold1-4,final_test}/{model}/
│   ├── model.pkl, metrics.json, y_pred.csv, run_config.json
│   └── shap_values.pkl, shap_importance.csv, shap_weekly_group.json,
│       shap_bar.png, shap_beeswarm.png, shap_config.json
└── stats/
    ├── hpo_comparison.csv
    ├── performance_summary.csv, performance_aggregated.csv
    ├── shap_importance_per_fold.csv, feature_importance_stability.csv
    ├── weekly_group_per_fold.csv, weekly_group_stability.csv
    ├── summary.md
    └── plots/{A,B,C}_feature_trend.png
```
