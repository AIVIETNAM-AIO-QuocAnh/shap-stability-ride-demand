# Module Heatmap: dữ liệu địa lý NYC taxi zone

Thư mục dữ liệu tĩnh, **không có code**. Chứa ranh giới địa lý và bảng tra cứu của 263 taxi
zone NYC, dùng làm nền cho bản đồ choropleth ở dashboard.

Đây là dữ liệu tham chiếu tải về từ nguồn công khai của NYC TLC, **không phải artifact do
pipeline sinh ra**, nên không bị xoá khi chạy lại experiment.

## Nội dung

```text
src/heatmap/taxi_zones/
├── taxi_zones.shp            1.5 MB   ranh giới polygon của 263 zone
├── taxi_zones.dbf             59 KB   thuộc tính đi kèm từng polygon
├── taxi_zones.shx            2.2 KB   chỉ mục của .shp
├── taxi_zones.prj            565 B    hệ toạ độ, dạng WKT
├── taxi_zones.cpg              5 B    encoding của .dbf (UTF-8)
├── taxi_zone_lookup.csv       12 KB   265 dòng: LocationID -> tên zone, borough
└── taxi_zone_map_*.jpg       6.9 MB   5 bản đồ tham khảo, mỗi borough một ảnh
```

Tổng 8.4 MB, trong đó riêng 5 ảnh JPG đã chiếm 6.9 MB.

## Shapefile

| Thuộc tính | Giá trị |
|---|---|
| Số record | 263 |
| Loại hình học | POLYGON |
| Field | `OBJECTID`, `Shape_Leng`, `Shape_Area`, `zone`, `LocationID`, `borough` |
| Khoá nối với dữ liệu model | `LocationID`, tương ứng `pu_location_id` trong feature table |
| Encoding | UTF-8 (khai báo ở `.cpg`) |

**Hệ toạ độ là điều quan trọng nhất cần biết ở đây.** `.prj` khai báo
`NAD_1983_StatePlane_New_York_Long_Island_FIPS_3104_Feet`, đơn vị **US survey foot**, không phải
lat/lon. Bounding box nằm trong khoảng x 913.175 tới 1.067.383 và y 120.122 tới 272.844, đúng
dạng toạ độ StatePlane chứ không phải độ.

Mọi thư viện bản đồ web đều cần WGS84 (EPSG:4326). Đưa thẳng shapefile này vào một thư viện bản
đồ mà không reproject thì **bản đồ trắng trơn và không báo lỗi gì**, vì toạ độ rơi ra ngoài phạm
vi hợp lệ của lat/lon.

Một số zone gồm nhiều mảnh rời (đảo, bãi bồi, phần đất bị nước cắt ngang); zone nhiều mảnh nhất
có **33 mảnh**. Khi chuyển sang GeoJSON phải xử lý bằng `MultiPolygon`, không thể coi mỗi record
là một polygon đơn.

## Bảng tra cứu

`taxi_zone_lookup.csv`, 265 dòng, cột `LocationID`, `Borough`, `Zone`, `service_zone`.

| Borough | Số zone |
|---|---|
| Queens | 69 |
| Manhattan | 69 |
| Brooklyn | 61 |
| Bronx | 43 |
| Staten Island | 20 |
| EWR (sân bay Newark) | 1 |
| Unknown | 1 |

**Lookup có 265 dòng nhưng shapefile chỉ có 263 polygon.** Hai id thừa là `264` (`Unknown`) và
`265` (`Outside of NYC`), đây là mã kỹ thuật của TLC cho chuyến không xác định được zone, đúng
theo thiết kế là không có ranh giới địa lý. Ngược lại không có polygon nào thiếu trong lookup.

## Quan hệ với phần còn lại của project

Chỉ [src/dashboard/build_zone_geojson.py](../dashboard/build_zone_geojson.py) đọc thư mục này.
Nó chạy **một lần** để:

1. đọc geometry bằng `pyshp`;
2. reproject từ StatePlane feet sang WGS84 bằng `pyproj`, lấy hệ nguồn từ chính `.prj`;
3. lọc còn đúng 50 frozen zone của experiment;
4. ghi ra `src/dashboard/zones_50.geojson` (0.27 MB).

Dashboard lúc chạy chỉ đọc file GeoJSON đó, **không cần thư viện GIS nào**.

```bash
conda activate shap-stability-ride-demand
python -m src.dashboard.build_zone_geojson
```

Đã kiểm: cả 50 frozen zone đều có geometry trong shapefile. Phân bố của chúng là Manhattan 25,
Brooklyn 15, Queens 9, Bronx 1; không zone nào thuộc Staten Island hay EWR lọt vào top 50, nên
bản đồ dashboard không phủ hai vùng đó.

Kiểm chứng sau khi reproject: toàn bộ 50 zone nằm trong lon -74.01 tới -73.75 và lat 40.62 tới
40.85, đúng phạm vi NYC.

## Năm ảnh JPG

`taxi_zone_map_bronx.jpg` và bốn ảnh cùng bộ là bản đồ tham khảo do TLC phát hành, để tra tay
xem một zone nằm ở đâu. **Không file code nào trong project đọc chúng.** Giữ lại vì tiện đối
chiếu khi kiểm tra kết quả bằng mắt; xoá đi thì dashboard vẫn chạy bình thường và tiết kiệm
6.9 MB.

## Nguồn

NYC Taxi and Limousine Commission, trang Trip Record Data, mục Taxi Zone Shapefile và Taxi Zone
Lookup Table. Cùng nguồn với dữ liệu HVFHV mà `src/data/download_raw.py` tải về.
