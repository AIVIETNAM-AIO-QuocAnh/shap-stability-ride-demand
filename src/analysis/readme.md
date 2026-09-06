# Module Analysis: `run_stats.py` (tổng hợp cuối, bàn giao)

Phần cuối của vai trò **AI Model** theo `m03-proposal.pdf` mục 4: tổng hợp `results/` (30 tổ hợp
variant × fold × model, đã có sẵn từ `src/pipeline/`, xem [../pipeline/README.md](../pipeline/README.md))
thành bảng số + nhận xét để bàn giao. Không train lại gì, chỉ đọc lại artifact đã có.

## Phạm vi: chỉ các chỉ số được proposal quy định

Module này tính **đúng và chỉ** những chỉ số nằm trong proposal:

| Nguồn trong proposal | Chỉ số |
|---|---|
| Mục 2.5 | MAE, RMSE, WAPE trên toàn bộ evaluation rows của từng fold |
| Mục 2.5 | mean ± **sample** standard deviation của 3 metric trên qua Fold 1-4 |
| Mục 2.5 | `final_test` (tháng 12) báo cáo **riêng**, không gộp vào mean/std |
| Mục 2.6 | `I_{j,f}` = mean \|φ\| của feature j trên 5.000 sample row của fold f |
| Mục 2.6 | mean ± sample standard deviation của `I_{j,f}` qua Fold 1-4 |
| Mục 2.6 | `I_weekly,f` = tổng `I_{j,f}` của weekly feature trong variant (A=3, B=1, C=2 feature) |
| Mục 2.6 | mean ± sample standard deviation của `I_weekly,f` qua Fold 1-4 |
| Mục 2.2 | Pearson correlation của 3 cặp weekly lag, mean và std trên 50 zone (đọc lại từ tầng Data) |

**Những gì cố ý KHÔNG tính.** Proposal mục 2.6 ghi rõ *"Không dùng coefficient of variation hoặc
significance test trong core scope"*, và mục 6 ghi *"Không cần thêm model, dataset, statistical
test, deployment hoặc experiment ngoài phạm vi trên để coi core project là hoàn thành"*. Vì vậy
module không tính:

- **coefficient of variation** (std/mean): proposal cấm trực tiếp;
- **rank / Spearman stability**: thước đo ngoài spec;
- **paired comparison kèm standard error** (`mean/SE`): mang tính significance test;
- **z-score của `final_test`** so với phân bố Fold 1-4: mang tính significance test;
- **trend / confound check** theo train size: experiment ngoài core matrix;
- **sensitivity của công thức group importance** (`mean_i |Σ_j φ_ij|`): proposal đã cố định công
  thức là `Σ_j mean_i |φ_ij|`.

Nếu cần các thước đo trên thì chúng thuộc phần follow-up ngoài core scope, và phải được ghi rõ là
ngoài spec, không đưa tự động vào bảng bàn giao.

**Hai ràng buộc đọc số từ proposal:**

1. **`final_test` tách riêng** (mục 2.5). Mọi hàm `compute_*` chỉ dùng Fold 1-4; `final_test` xuất
   hiện như cột riêng (`*_final_test`), không lẫn vào con số ổn định.
2. **std đọc cùng mean** (mục 2.6): *"không kết luận một feature 'ổn định hơn' chỉ dựa vào standard
   deviation nếu mức mean importance khác nhau quá lớn"*. `summary.md` in cảnh báo này kèm khoảng
   mean thực tế quan sát được.

## `RunStats`

Đọc lại toàn bộ `results/{A,B,C}/{fold1-4,final_test}/{xgboost,lightgbm}/`. Chạy độc lập sau khi
`python run.py all` đã xong (xem [../pipeline/README.md](../pipeline/README.md)).

| Method | Việc làm | Output |
|---|---|---|
| `load_correlation_summary` | đọc lại correlation 3 cặp weekly lag do tầng Data sinh (proposal mục 2.2) | (nội bộ) |
| `load_metrics_df` | gom `metrics.json` của 30 tổ hợp | `performance_summary.csv` (raw, theo fold) |
| `load_shap_long_df` | gom `shap_importance.csv`, lọc đúng `weekly_features` theo variant | (nội bộ) |
| `compute_performance_summary` | mean/std MAE/RMSE/WAPE qua Fold 1-4 + cột final_test | `performance_aggregated.csv` |
| `compute_feature_stability` | mean/std `I_{j,f}` qua Fold 1-4 theo (variant, model, feature) + cột final_test | `feature_importance_stability.csv` |
| `compute_group_stability` | đọc `shap_weekly_group.json`, mean/std qua Fold 1-4 + cột final_test | `weekly_group_stability.csv` |
| `plot_feature_trends` | line plot `I_{j,f}` theo fold (gồm cả `final_test` để xem, không dùng tính std), 1 file/variant | `plots/{variant}_feature_trend.png` |
| `write_summary_md` | correlation, rồi bảng mean ± std cho prediction, feature-level và group-level; trả lời **cả ba** tiêu chí hoàn thành ở mục 6 | `summary.md` |
| `run` | orchestrate toàn bộ theo đúng thứ tự trên | (không ghi file) |

**Lưu ý về `compute_group_stability`:** số feature trong nhóm khác nhau giữa các variant (A=3, B=1,
C=2), nên bảng này **không so sánh trực tiếp được giữa các variant**. Riêng variant B nhóm chỉ có 1
feature nên group importance ≡ feature importance. `summary.md` in cảnh báo này tự động dựa trên
`variant_map`.

**Quan hệ với `src/pipeline/summarize.py`:** ba bảng `performance_aggregated.csv`,
`feature_importance_stability.csv` và `weekly_group_stability.csv` được cả hai module ghi. Sau khi
Analysis được thu về đúng spec, hai bên dùng **cùng định nghĩa và cùng schema**, nên nội dung trùng
khớp từng giá trị; thứ tự chạy đúng là `run.py all` trước, `run_stats` sau.

Cột `fold` luôn ép về `pd.Categorical(..., categories=folds, ordered=True)` ngay sau khi tạo
DataFrame, tránh pandas sort alphabet (`"final_test" < "fold1"` theo string, sẽ đảo sai thứ tự
thời gian nếu không ép). Mọi `groupby` theo sau dùng `observed=True` để không sinh tổ hợp rỗng.

`self.stats_folder` và `self.plots_folder` (con của `stats_folder`) được `mkdir` 1 lần trong
`__init__`, các method ghi file chỉ dùng lại `self.stats_folder` / `self.plots_folder`, không tạo
thư mục rải rác trong từng method.

## Cấu trúc `results/stats/`

```text
results/stats/
├── hpo_comparison.csv               # baseline vs tuned trên HPO split (ghi bởi pipeline)
├── performance_summary.csv          # raw: variant, model, fold, mae, rmse, wape
├── performance_aggregated.csv       # variant, model, {mae,rmse,wape}_{mean,std}, {mae,rmse,wape}_final_test
├── shap_importance_per_fold.csv     # I_{j,f} theo từng fold (ghi bởi pipeline)
├── feature_importance_stability.csv # variant, model, feature, mean/std importance, importance_final_test
├── weekly_group_per_fold.csv        # I_weekly,f theo từng fold (ghi bởi pipeline)
├── weekly_group_stability.csv       # variant, model, mean/std group importance, group_importance_final_test
├── summary.md                       # bàn giao: 5 mục, trả lời cả ba tiêu chí hoàn thành mục 6
└── plots/
    └── {A,B,C}_feature_trend.png
```

Correlation giữa ba weekly lag (proposal mục 2.2) thuộc tầng Data, module này chỉ **đọc lại** để
`summary.md` trả lời được cả ba tiêu chí hoàn thành thay vì trỏ sang file khác. Nguồn:
`data/processed/correlation_summary.csv` và `correlation_by_zone.csv`.

Cấu trúc `summary.md`: (1) correlation, (2) prediction performance, (3) SHAP feature-level,
(4) weekly-group, (5) trả lời ba câu hỏi ở proposal mục 6.

## Cách chạy

```bash
python -m src.analysis.run_stats
```

Chạy bằng `-m` (module) từ project root, không chạy trực tiếp `python src/analysis/run_stats.py`.
Lý do: `src/` không có `__init__.py` (namespace package), file dùng `from src.load_config import ...`
nên cần project root nằm trong `sys.path`, chỉ có khi chạy qua `-m`.
