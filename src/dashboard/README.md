# Module Dashboard: visualize dự báo demand theo zone và giờ

Dashboard Streamlit đọc lại prediction artifact trong `results/`. Không train lại, không sửa gì
trong core pipeline. Đây là công cụ trình bày, nằm **ngoài core scope** của `m03-proposal.pdf`.

## Cách chạy

```bash
source .venv/bin/activate

# chỉ cần chạy một lần, sinh zones_50.geojson
pip install pyshp pyproj
python -m src.dashboard.build_zone_geojson

streamlit run dashboard_app.py
```

`pyshp` và `pyproj` chỉ phục vụ build step. Hai package này **không** được thêm vào
`environment.yaml` vì file đó khoá môi trường của core experiment theo proposal; dashboard là
phần mở rộng nên khai báo dependency riêng ở đây.

## Bố cục: một khung hình, không cuộn

Mọi control nằm **trên màn hình chính**, không dùng sidebar. Lý do: sidebar có thể đang ở trạng
thái thu gọn, người dùng mở app lên sẽ tưởng dashboard không cho chọn gì.

Control được đặt cạnh đúng thứ nó tác động, thay vì dồn hết lên đầu trang:

```text
Ride demand forecast by zone and hour
──────────────────────────────────────────────────────────────────────
🎄 Christmas Day · Thursday, 25 Dec 2025 · ...   [fold] [n zones] [variant/model]
┌──────────┬──────────────────────┬─────────────────────────┐
│ Date     │ [Map layer]          │ [Top N]                 │
│ Hour     │                      │  bảng Top N zone        │
│ District │  Choropleth 50 zone  │                         │
│ Zone     │                      │  biểu đồ 24 giờ         │
│ Variant  │                      │                         │
│  chú thích                      │                         │
│ Model    │                      │                         │
└──────────┴──────────────────────┴─────────────────────────┘
```

Giao diện hoàn toàn bằng **tiếng Anh**; comment và docstring giữ tiếng Việt cho đồng bộ với
các module khác trong project.

**Slicer dữ liệu gom vào pane hẹp bên trái** (Date, Hour, District, Zone, Variant, Model): cột
dọc lấy chỗ theo chiều ngang, vốn còn thừa trên màn rộng, thay vì chiều dọc vốn là thứ khan hiếm.

**`Map layer` và `Top N` không nằm trong pane filter** mà đặt ngay trên bản đồ và bảng: chúng chỉ
đổi cách hiển thị một thành phần cụ thể, để cạnh thành phần đó thì dễ nối hơn.

Badge ngày và các chip trạng thái (fold, số zone, variant/model) nằm **cùng một hàng** ngay dưới
tiêu đề.

**Badge phải bọc trong một `div` block-level dùng `display:flex`.** Nếu để pill ở dạng
`inline-flex` trần thì nó là hộp inline: chiều cao hộp lớn hơn line box của thẻ cha nên tràn ra
ngoài và **đè lên bản đồ** ngay bên dưới. Lỗi này đã gặp hai lần, một lần do `line-height` chật,
một lần do `inline-flex` trần. Chú thích variant nằm **ngay dưới slicer Variant**, để nối được ngay chú thích với cái vừa bấm.
Danh sách feature lấy động từ config nên không thể lệch với dữ liệu model thật sự dùng.

Chú thích gồm đúng **hai dòng ngắn**, mỗi dòng phải nằm gọn một hàng trong pane hẹp:

| Variant | Dòng mô tả | Dòng feature | Bề rộng dài nhất |
|---|---|---|---|
| A | Three lags kept separate | `lag_168 · lag_336 · lag_504` | ~177px |
| B | Median of the three lags | `median_lag_3w` | ~138px |
| C | Nearest lag plus median | `lag_168 · median_lag_3w` | ~150px |

Pane filter rộng 19% chiều ngang, tức khoảng 236px khả dụng trên màn 1366px, nên dòng dài nhất
vẫn dư 59px. **Viết mô tả dài hơn khoảng 26 ký tự là bị ngắt dòng giữa chừng, trông rất xấu**;
câu giải thích đầy đủ đặt ở tooltip của chính slicer Variant thay vì kéo dài chú thích.

Chú thích dựng bằng HTML chứ không dùng `st.caption`: cần kiểm soát `line-height` và
`margin-bottom` để chữ không dính sát nhãn `Model` ngay bên dưới. Hàng trên cùng chỉ còn các bộ lọc trả lời câu hỏi "đang xem dữ liệu
nào", nhờ vậy gom được về một hàng thay vì hai.

Biểu đồ 24 giờ **luôn hiển thị**, không nằm trong tab. Nó cộng dồn theo giờ trên **toàn bộ zone
đang chọn**, nên luôn khớp đúng phạm vi mà District và Zone đang mô tả:

| Bộ lọc | Tiêu đề biểu đồ |
|---|---|
| không lọc gì | All 50 selected zones, total demand over 24 hours |
| District = Queens | All 9 selected zones, total demand over 24 hours |
| District = Bronx | Mott Haven/Port Morris over 24 hours |
| Zone = JFK Airport | JFK Airport over 24 hours |

Bản trước khoá cứng biểu đồ vào zone đứng đầu bảng Top N, khiến nó luôn hiện JFK Airport bất kể
người dùng lọc gì. Đó là lỗi thiết kế: biểu đồ tự chọn hộ phạm vi thay vì bám bộ lọc.

### Đường weekly lag vẽ chồng

Biểu đồ vẽ thêm **weekly feature của chính variant đang chọn**, đọc từ file fold mà Pipeline đã
dùng để train (`data/folds/{fold}/{val,test}.csv`) nên đúng bằng giá trị model nhìn thấy:

| Variant | Đường thêm vào |
|---|---|
| A | `lag_168`, `lag_336`, `lag_504` |
| B | `median_lag_3w` |
| C | `lag_168`, `median_lag_3w` |

Chúng vẽ bằng **nét đứt, mảnh (1.6px), opacity 0.55, màu xám nhạt**, không có marker; còn
`Forecast` và `Actual` giữ nét liền 2.4px kèm marker. Nhờ vậy đọc được model đang bám theo tuần
nào mà mắt vẫn dừng ở hai đường chính.

Kiểm chứng: tổng theo giờ của `lag_168` khớp từng dòng với phép tính độc lập từ `fold1/val.csv`,
và cùng độ lớn với actual (đỉnh 22.623 so với 21.010 lúc 22h ngày 01/08).

**Legend đặt dưới vùng vẽ, không đặt trên.** Tiêu đề biểu đồ và legend nếu cùng nằm trong dải
margin phía trên sẽ đâm vào nhau khi tiêu đề dài. Đặt legend xuống dưới thì vị trí của nó độc lập
hoàn toàn với độ dài tiêu đề và với số mục (3 mục ở variant B, 5 mục ở variant A).

Nhãn trục x (`hour`) đã bỏ vì các mốc 0..24 tự rõ nghĩa, lấy chỗ đó cho legend.

**Margin dưới phải đủ chứa cả nhãn mốc lẫn legend.** Plotly không tự nới margin đã khai báo tường
minh, legend đặt ngoài vùng vẽ mà margin thiếu là bị cắt. Với `PROFILE_HEIGHT = 254`, margin trên
28 và dưới 54, vùng vẽ còn 172px; legend ở `y = -0.16` nằm trong khoảng 28px tới 48px dưới đáy
vùng vẽ, còn dư 6px so với margin 54px. Đổi `PROFILE_HEIGHT` thì tính lại tỉ lệ này.

### Bảng Top N

Hai cột `dự báo` và `thực tế` vẽ thành bar ngay trong cell bằng `st.column_config.ProgressColumn`,
**dùng chung một thang** (max của cả hai cột trong các dòng đang hiện), nên so được cả giữa các
zone lẫn giữa dự báo với thực tế.

Cột `sai số` đổi màu theo dấu bằng `pandas.Styler`: **đỏ khi dương** (dự báo cao hơn thực tế),
**xanh khi âm** (dự báo thấp hơn). Styler và `column_config` dùng chung được trong cùng một
`st.dataframe`.

### Chiều cao và khoảng hở đầu trang

`.block-container` phải có `padding-top` tối thiểu **4rem**. Streamlit vẽ một thanh header cố
định cao 3.75rem; hạ padding xuống nhỏ hơn thì hàng đầu tiên chui xuống dưới header và **bị che**
(lỗi đã gặp: banner ngày lễ bị cắt mất nửa trên).

Chiều cao các thành phần cố định bằng `MAP_HEIGHT`, `TABLE_HEIGHT`, `PROFILE_HEIGHT`; công thức
cân bằng hai cột ghi trong comment ngay trên chúng. Tổng hiện tại khoảng **702px**:

| Màn hình | Vùng hiển thị | Kết quả |
|---|---|---|
| 1920x1080 | ~960px | vừa, dư 258px |
| 1440x900 | ~780px | vừa, dư 78px |
| 1366x768 | ~648px | cuộn 54px, hạ `MAP_HEIGHT` và `PROFILE_HEIGHT` nếu cần vừa hẳn |

## Ánh xạ ngày sang fold

Mỗi fold đánh giá đúng một tháng, ghép lại thành dải liên tục:

| fold1 | fold2 | fold3 | fold4 | final_test |
|---|---|---|---|---|
| 8/2025 | 9/2025 | 10/2025 | 11/2025 | 12/2025 |

Đọc động từ `configs/data.yaml` (`splits.*.eval_start` và `eval_end_exclusive`), không hardcode.
Split `hpo` bị loại vì chỉ dùng để tune, không nằm trong 30 core run.

## Nguồn dữ liệu

**Prediction:** `results/{A,B,C}/{fold1..fold4,final_test}/{xgboost,lightgbm}/y_pred.csv`,
30 file, mỗi file 50 zone nhân số giờ trong tháng. Schema đã có sẵn cả `y_true` lẫn `y_pred` nên
tính được metric mà không cần load lại model.

**Bản đồ:** `src/heatmap/taxi_zones/taxi_zones.shp`, chi tiết ở
[../heatmap/README.md](../heatmap/README.md). Shapefile ở hệ
`NAD83 StatePlane New York Long Island (feet)`, không phải lat/lon, nên `build_zone_geojson.py`
reproject sang WGS84 bằng `pyproj` trước khi ghi ra GeoJSON. Kiểm chứng sau khi convert: toàn bộ
50 zone nằm trong lon -74.01 đến -73.75, lat 40.62 đến 40.85, đúng phạm vi NYC.

**Ngày lễ:** `USFederalHolidayCalendar` của pandas, đúng lịch mà `src/data/build_panel.py` dùng để
sinh feature `is_holiday`. Dùng chung nguồn nên banner trên dashboard khớp với thứ model thực sự
nhìn thấy lúc train.

## Metric

Dashboard **không hiển thị** MAE, RMSE, WAPE nữa. Sai số được đọc trực tiếp trên bản đồ (lớp
"Sai số") và trên cột `sai số` của bảng, có màu theo dấu.

`data_access.compute_metrics()` vẫn còn và vẫn dùng nguyên hàm của `src/pipeline/metrics.py`,
nên muốn hiện lại chỉ cần gọi nó. Số nó trả về khớp từng chữ số với `results/stats/`: variant A,
fold1, xgboost cả tháng cho MAE 20.39, RMSE 33.90, WAPE 8.78, đúng bằng
`results/A/fold1/xgboost/metrics.json`.

## Vì sao toggle thay vì chồng hai lớp

Yêu cầu ban đầu là chồng heatmap actual lên bản đồ. Hai lớp choropleth chồng nhau thì lớp trên
che kín lớp dưới, không đọc được gì. Thay bằng toggle 3 chế độ trên cùng một bản đồ:

- **Dự báo** và **Thực tế** dùng **chung một thang màu** (0 đến max của cả hai lớp), nên đổi qua
  lại là so sánh được trực tiếp bằng mắt.
- **Sai số** (`y_pred - y_true`) dùng thang diverging đối xứng quanh 0. Đây thực chất là cách
  "chồng" duy nhất đọc được, vì nó nén hai lớp thành một đại lượng.

Tooltip của cả 3 chế độ đều hiện đủ ba số: dự báo, thực tế, sai số.

## Hai trang

| Trang | Module | Nội dung |
|---|---|---|
| **Forecast** | `src/dashboard/page_forecast.py` | bản đồ demand theo zone và giờ, gói gọn một khung hình |
| **Experiment** | `src/dashboard/page_experiment.py` | so sánh kết quả thí nghiệm theo đúng cấu trúc README gốc |

Điều hướng bằng `st.navigation` với callable, **không** dùng thư mục `pages/`: các trang là module
trong `src/dashboard/` nên không phải lo `sys.path` của từng file trang.

Phải đặt `url_path` tường minh cho mỗi `st.Page`: hai trang cùng tên hàm `render` nên Streamlit suy
ra cùng một pathname và báo lỗi trùng URL.

### Trang Experiment

Chỉ đọc lại `results/stats/` và `data/processed/correlation_summary.csv`, **không tính lại** chỉ số
nào, nên số trên trang luôn khớp báo cáo. Đã kiểm: MAE mean, SHAP mean và correlation trên trang
khớp từng chữ số với file gốc.

Năm mục bám đúng cấu trúc README:

1. Correlation giữa ba weekly lag (proposal mục 2.2)
2. SHAP feature importance mean ± std qua Fold 1-4, kèm hình xu hướng theo fold (mục 2.6)
3. Weekly-group importance mean ± std (mục 2.6)
4. MAE/RMSE/WAPE mean ± std, chọn được metric, kèm bảng chênh lệch so với baseline A (mục 2.5)
5. Baseline vs tuned trên HPO split (mục 3.2)

Không có coefficient of variation, rank stability hay significance test, đúng phần Phạm vi báo cáo
của README. Trang này là báo cáo nên cho phép cuộn, khác trang Forecast.

## Cấu trúc

```text
dashboard_app.py              # entry point + điều hướng, nằm ở PROJECT ROOT (lý do ở dưới)

src/dashboard/
├── README.md                 # file này
├── build_zone_geojson.py     # build step: shapefile StatePlane -> GeoJSON WGS84, lọc 50 zone
├── data_access.py            # đọc + cache prediction, geojson, zone lookup, lịch lễ, bảng stats
├── holidays_theme.py         # theme riêng cho 11 ngày lễ liên bang
├── page_forecast.py          # trang Forecast
├── page_experiment.py        # trang Experiment
└── zones_50.geojson          # sinh ra từ build step, 0.27 MB
```

### Vì sao `dashboard_app.py` ở project root

`streamlit run <script>` đặt **thư mục chứa script** lên `sys.path`, không phải thư mục hiện
hành. Để file trong `src/dashboard/` thì chỉ `src/dashboard/` nằm trên path, nên mọi
`from src.… import` đều hỏng với `ModuleNotFoundError: No module named 'src'`, kể cả khi đã
`cd` về project root.

Đặt script ở root thì project root lên path, dùng chung được `src/load_config.py`,
`src/pipeline/metrics.py` và `src/utilities.py` với core pipeline. Phần logic nặng vẫn nằm ở
`src/dashboard/data_access.py`; file ở root chỉ dựng giao diện.

Lưu ý khi test: `streamlit.testing.v1.AppTest` **không** tái hiện `sys.path` này (nó chạy với
project root sẵn trên path), nên AppTest pass không đảm bảo `streamlit run` chạy được. Muốn chắc
thì phải chạy `streamlit run` thật.

## Hiệu năng

`@st.cache_data` khoá theo `(variant, fold, model)` nên đổi ngày, giờ, zone hay lớp bản đồ trong
cùng một tổ hợp thì không đọc lại file. Chỉ đọc file của tổ hợp đang chọn, không nạp trước cả 30
file. GeoJSON và zone catalog cache riêng, đọc một lần.
