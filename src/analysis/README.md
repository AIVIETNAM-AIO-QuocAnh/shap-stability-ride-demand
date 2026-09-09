# Module Analysis

Module Analysis đọc các artifact do Pipeline và Data sinh ra, sau đó tạo bảng, hình và báo cáo tổng
hợp. Module này không train model và không thay đổi experiment protocol.

## Input và output

Analysis cần đủ kết quả trong:

```text
results/{A,B,C}/{fold1,fold2,fold3,fold4,final_test}/{xgboost,lightgbm}/
data/processed/correlation_summary.csv
```

Module tạo các bảng prediction performance, SHAP feature importance, weekly-group importance, HPO
comparison và hình trend theo fold. Báo cáo chính là `results/stats/summary.md`.

## Chỉ số

| Nhóm | Nội dung |
|---|---|
| Prediction | MAE, RMSE, WAPE theo fold; `final_test` báo cáo riêng. |
| Feature SHAP | Mean absolute SHAP trên 5.000 sample rows/fold. |
| Feature stability | Mean và sample standard deviation qua Fold 1-4. |
| Weekly group | Tổng importance của weekly features theo variant. |
| Correlation | Pearson correlation của ba cặp weekly lag trên 50 zones. |

## Cách chạy

Chạy từ project root sau khi `python run.py all` hoàn tất:

```bash
python -m src.analysis.run_stats
```

Kết quả chi tiết nằm trong [`results/stats/summary.md`](../../results/stats/summary.md).
