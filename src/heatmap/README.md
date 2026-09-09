# Module Heatmap

Thư mục này chứa geometry và lookup table của NYC taxi zones cho bản đồ Dashboard.

## Files

```text
src/heatmap/taxi_zones/
├── taxi_zones.shp/.shx/.dbf/.prj/.cpg
├── taxi_zone_lookup.csv
└── taxi_zone_map_*.jpg
```

Shapefile có 263 polygon; lookup table ánh xạ `LocationID` với tên zone và borough. Model key
`pu_location_id` tương ứng với `LocationID`.

Shapefile dùng NAD83 StatePlane New York Long Island (feet), không phải latitude/longitude. Script
dashboard reproject sang WGS84 (EPSG:4326) và lọc 50 frozen zones của experiment.

## Build GeoJSON

Chạy từ project root sau khi frozen zones đã được tạo:

```bash
python -m src.dashboard.build_zone_geojson
```

Script ghi `src/dashboard/zones_50.geojson`, là file Dashboard sử dụng khi hiển thị bản đồ.
