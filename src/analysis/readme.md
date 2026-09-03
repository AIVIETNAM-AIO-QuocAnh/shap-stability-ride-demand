# Module Analysis — `run_stats.py` (tổng hợp cuối, bàn giao)

Phần cuối của vai trò **AI Model** theo `m03-proposal.pdf` mục 4: tổng hợp `results/` (30 tổ hợp
variant × fold × model, đã có sẵn từ `src/pipeline/`, xem [../pipeline/readme.md](../pipeline/readme.md))
thành bảng số + nhận xét để bàn giao. Không train lại gì, chỉ đọc lại artifact đã có.

## `RunStats`

Đọc lại toàn bộ `results/{A,B,C}/{fold1-4,final_test}/{xgboost,lightgbm}/`. Chạy độc lập sau khi
`python run.py` đã xong (xem [../pipeline/readme.md](../pipeline/readme.md) mục "Cách chạy").

**Hai ràng buộc từ proposal quyết định cách code báo cáo:**

1. **`final_test` tách riêng** (mục 2.5: "Final test tháng 12 được báo cáo riêng"; mục 2.6: "Qua 4
   fold, báo cáo mean và sample standard deviation"). Mọi hàm `compute_*` chỉ dùng `fold1-4`;
   `final_test` xuất hiện như cột riêng (`*_final_test`), không lẫn vào con số ổn định.
2. **Không dùng CV để kết luận** (mục 2.6: *"Không dùng coefficient of variation hoặc significance
   test trong core scope"*, kèm cảnh báo không so std giữa các feature có mean chênh lệch lớn).
   **Số liệu chính theo spec chỉ là mean ± std.** Cột `cv` vẫn được tính và lưu, nhưng nằm ở mục
   Diagnostic của `summary.md` và ghi rõ là ngoài spec — trên dữ liệu này Spearman giữa
   `mean_importance` và `cv` là **−0.839**, tức CV chủ yếu phản ánh độ lớn của mean chứ không phải
   độ bất ổn.

   Rank stability (`compute_rank_stability`) là thước đo **bổ sung do nhóm thêm vào**, không nằm
   trong spec — nó chỉ không bị cấm. Trên dữ liệu này nó **đụng trần** (variant A đạt Spearman
   1.0000) nên không phân biệt được A/B/C; `summary.md` ghi rõ điều đó thay vì diễn giải "không có
   khác biệt".

| Method | Việc làm | Output |
|---|---|---|
| `load_metrics_df` | gom `metrics.json` của 30 tổ hợp | `performance_summary.csv` (raw, theo fold) |
| `load_shap_long_df` | gom `shap_importance.csv`, lọc đúng `weekly_features` theo variant | (nội bộ) |
| `load_shap_full_df` | như trên nhưng **không lọc** — giữ toàn bộ feature, cần cho rank stability | (nội bộ) |
| `compute_performance_summary` | mean/std MAE/RMSE/WAPE qua fold1-4 + cột final_test | `performance_aggregated.csv` |
| `compute_paired_comparison` | so sánh MAE theo từng cặp (fold × model) giữa B/C và baseline A — cùng fold/model/row nên triệt tiêu nhiễu theo mùa. Báo cả std lẫn **SE = std/√n** (claim về *trung bình* thì mẫu số là SE), và tách riêng từng model vì 8 cặp gộp không độc lập | `paired_mae_comparison.csv` |
| `compute_feature_stability` | mean/std importance qua fold1-4 theo (variant, model, feature) + cột final_test (kèm `cv` diagnostic) | `feature_importance_stability.csv` |
| `compute_group_stability` | đọc `shap_weekly_group.json`, mean/std qua fold1-4 + cột final_test (kèm `cv` diagnostic) | `weekly_group_stability.csv` |
| `compute_final_test_shift` | z-score của `final_test` so với phân bố fold1-4 của chính feature đó — fold1-4 đều là tháng đồng dạng nên đây là phép thử regime khác | `final_test_shift.csv` |
| `compute_rank_stability` | Spearman giữa các cặp fold, tách 2 cột: toàn bộ feature vs **chỉ non-zone** (50/58 là zone dummy, làm phồng Spearman lên ~0.98). Kèm `importance_swing_pct` để lộ giới hạn: rank không nhạy biên độ. **Không nằm trong spec proposal**, chỉ là thước đo bổ sung không bị cấm | `rank_stability.csv` |
| `compute_group_sensitivity` | đọc `shap_values.pkl`, so công thức group của proposal (`Σ_j mean_i \|φ_ij\|`) với công thức thay thế (`mean_i \|Σ_j φ_ij\|`) để đo mức triệt tiêu dấu giữa feature tương quan | `group_formula_sensitivity.csv` |
| `plot_feature_trends` | line plot importance theo fold (gồm cả `final_test` để xem xu hướng, không dùng tính std), 1 file/variant | `plots/{variant}_feature_trend.png` |
| `plot_cv_comparison` | bar chart CV feature-level vs group-level (diagnostic, ngoài spec) | `plots/cv_comparison.png` |
| `write_summary_md` | bảng mean±std làm chính, paired comparison, rank stability, mục Diagnostic tách riêng cho CV + sensitivity, kết luận tự sinh có hạ giọng theo độ mạnh bằng chứng | `summary.md` |
| `run` | orchestrate toàn bộ theo đúng thứ tự trên | — |

**Lưu ý về `compute_group_stability`:** số feature trong nhóm khác nhau giữa các variant (A=3, B=1,
C=2), nên bảng này **không so sánh trực tiếp được giữa các variant**. Riêng variant B nhóm chỉ có 1
feature nên group importance ≡ feature importance (cột `cv` của 2 file trùng nhau đến từng chữ số).
`summary.md` có in cảnh báo này tự động dựa trên `variant_map`.

Cột `fold` luôn ép về `pd.Categorical(..., categories=folds, ordered=True)` ngay sau khi tạo
DataFrame — tránh pandas sort alphabet (`"final_test" < "fold1"` theo string, sẽ đảo sai thứ tự thời
gian nếu không ép). Mọi `groupby` theo sau dùng `observed=True` để không sinh tổ hợp rỗng.

`self.stats_folder` và `self.plots_folder` (con của `stats_folder`) được `mkdir` 1 lần trong
`__init__`, các method ghi file chỉ dùng lại `self.stats_folder` / `self.plots_folder`, không tạo
thư mục rải rác trong từng method.

## Cấu trúc `results/stats/`

```text
results/stats/
├── performance_summary.csv          # raw: variant, model, fold, mae, rmse, wape
├── performance_aggregated.csv       # variant, model, {mae,rmse,wape}_{mean,std}, {mae,rmse,wape}_final_test
├── paired_mae_comparison.csv        # variant, scope (gộp/từng model), n_pairs, mean/std/se_delta_mae, mean_over_se
├── feature_importance_stability.csv # variant, model, feature, mean/std importance (+ cv diagnostic)
├── weekly_group_stability.csv       # variant, model, mean/std group importance (+ cv diagnostic)
├── final_test_shift.csv             # z-score của final_test so với phân bố fold1-4
├── rank_stability.csv               # Spearman (toàn bộ + non-zone), ranks_by_fold, importance_swing_pct
├── group_formula_sensitivity.csv    # variant, model, group theo 2 công thức + cancellation_ratio
├── summary.md                       # bàn giao: bảng + đánh giá tự sinh
└── plots/
    ├── {A,B,C}_feature_trend.png
    └── cv_comparison.png            # diagnostic (ngoài spec)
```

## Cách chạy

```bash
python -m src.analysis.run_stats
```

Chạy bằng `-m` (module) từ project root, không chạy trực tiếp `python src/analysis/run_stats.py` —
`src/` không có `__init__.py` (namespace package), file dùng `from src.utilities import ...` nên cần
project root nằm trong `sys.path`, chỉ có khi chạy qua `-m`.
