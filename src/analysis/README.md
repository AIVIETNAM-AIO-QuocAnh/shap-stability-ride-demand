# Module Analysis: `run_stats.py`

Module Analysis đọc lại các artifacts đã sinh bởi Pipeline và tạo bảng, hình, cùng
`results/stats/summary.md`. Module này không train model, không tính lại raw data và không
thay đổi experiment protocol.

## Core metrics

| Phần | Chỉ số |
|---|---|
| Correlation | Pearson correlation của ba cặp weekly lag, mean và sample standard deviation trên 50 zones |
| Prediction | MAE, RMSE, WAPE trên toàn bộ evaluation rows của từng fold |
| Prediction summary | Mean ± sample standard deviation qua Fold 1-4; `final_test` tháng 12 tách riêng |
| Feature SHAP | `I_{j,f}` = mean |φ| của feature *j* trên 5,000 sample rows của fold *f* |
| Feature stability | Mean ± sample standard deviation của `I_{j,f}` qua Fold 1-4 |
| Weekly group | Tổng `I_{j,f}` của weekly features trong từng variant, sau đó tính mean ± sample standard deviation |

Core analysis không tính coefficient of variation, rank/Spearman stability, paired standard
error, z-score của `final_test`, trend/confound diagnostics hoặc sensitivity của group formula.
Các phép đo đó nằm ngoài core scope và không được tự động thêm vào bảng bàn giao.

Hai quy ước quan trọng:

1. `final_test` luôn là cột riêng (`*_final_test`), không được gộp vào Fold 1-4 mean/std.
2. Standard deviation phải được đọc cùng mean importance; không xếp hạng stability giữa các
   feature có mean importance chênh lệch lớn chỉ bằng std.

## `RunStats`

`RunStats` đọc `results/{A,B,C}/{fold1-4,final_test}/{xgboost,lightgbm}/` sau khi
`python run.py all` hoàn tất.

| Method | Việc làm | Output |
|---|---|---|
| `load_correlation_summary` | Đọc correlation summary do Data module sinh | Nội bộ |
| `load_metrics_df` | Gom `metrics.json` của 30 tổ hợp | `performance_summary.csv` |
| `load_shap_long_df` | Gom `shap_importance.csv`, lọc weekly features theo variant | Nội bộ |
| `compute_performance_summary` | Mean/std MAE/RMSE/WAPE qua Fold 1-4 + `final_test` | `performance_aggregated.csv` |
| `compute_feature_stability` | Mean/std `I_{j,f}` theo variant, model, feature + `final_test` | `feature_importance_stability.csv` |
| `compute_group_stability` | Đọc weekly-group artifacts và tính mean/std + `final_test` | `weekly_group_stability.csv` |
| `plot_feature_trends` | Vẽ `I_{j,f}` theo fold cho từng variant | `plots/{variant}_feature_trend.png` |
| `write_summary_md` | Ghi correlation, prediction, SHAP và weekly-group tables cùng kết luận | `summary.md` |

Group sizes khác nhau giữa variants (A=3, B=1, C=2), vì vậy weekly-group table không thể
được dùng để so sánh trực tiếp stability giữa các variants. Với Variant B, group importance
trùng feature importance vì group chỉ có một feature.

## Cấu trúc `results/stats/`

```text
results/stats/
├── hpo_comparison.csv
├── performance_summary.csv
├── performance_aggregated.csv
├── shap_importance_per_fold.csv
├── feature_importance_stability.csv
├── weekly_group_per_fold.csv
├── weekly_group_stability.csv
├── summary.md
└── plots/{A,B,C}_feature_trend.png
```

Correlation do Data module sinh ra và Analysis chỉ đọc lại từ
`data/processed/correlation_summary.csv` và `correlation_by_zone.csv`.

## Cách chạy

```bash
python -m src.analysis.run_stats
```

Chạy bằng `-m` từ project root. Trước đó cần có đầy đủ model, prediction và SHAP artifacts
từ Pipeline. Kết quả chi tiết nằm trong `results/stats/summary.md`.
