"""Chuyển taxi zone shapefile sang GeoJSON WGS84, chỉ giữ 50 frozen zone.

Chạy đúng một lần. Shapefile trong `src/heatmap/taxi_zones/` dùng hệ toạ độ
NAD83 StatePlane New York Long Island (đơn vị feet), trong khi mọi thư viện bản đồ web
đều cần lat/lon WGS84, nên phải reproject trước khi dùng.

Tách riêng khỏi `app.py` để dashboard lúc chạy không cần thư viện GIS nào:
app chỉ đọc file GeoJSON đã sinh sẵn.

    python -m src.dashboard.build_zone_geojson
"""

import json
from pathlib import Path

import pyproj
import shapefile

from src.load_config import PROJECT_ROOT, load_data_config
from src.utilities import load_frozen_zone_ids


SHAPEFILE_PATH = PROJECT_ROOT / "src" / "heatmap" / "taxi_zones" / "taxi_zones"
PROJECTION_PATH = SHAPEFILE_PATH.with_suffix(".prj")
OUTPUT_PATH = PROJECT_ROOT / "src" / "dashboard" / "zones_50.geojson"
WGS84 = "EPSG:4326"


def _build_transformer() -> pyproj.Transformer:
    """Tạo transformer từ hệ toạ độ ghi trong .prj sang WGS84."""
    projection_text = PROJECTION_PATH.read_text(encoding="utf-8").strip()
    source_crs = pyproj.CRS.from_wkt(projection_text)
    return pyproj.Transformer.from_crs(source_crs, WGS84, always_xy=True)


def _transform_ring(ring, transformer: pyproj.Transformer) -> list[list[float]]:
    """Reproject một vòng toạ độ và trả về [lon, lat] theo thứ tự GeoJSON."""
    x_values = [point[0] for point in ring]
    y_values = [point[1] for point in ring]
    longitudes, latitudes = transformer.transform(x_values, y_values)
    return [[round(lon, 6), round(lat, 6)] for lon, lat in zip(longitudes, latitudes)]


def _shape_to_polygons(shape, transformer: pyproj.Transformer) -> list[list[list[list[float]]]]:
    """Tách shapefile record thành danh sách polygon đã reproject.

    Shapefile gộp mọi vòng của một record vào một mảng điểm phẳng, phân tách bằng
    `shape.parts`. Mỗi vòng được coi là một polygon riêng (MultiPolygon), cách xử lý
    này đủ đúng cho taxi zone vì các zone không có lỗ thủng bên trong.
    """
    part_starts = list(shape.parts) + [len(shape.points)]
    polygons = []
    for start, end in zip(part_starts, part_starts[1:]):
        ring = shape.points[start:end]
        if len(ring) < 4:
            continue
        transformed = _transform_ring(ring, transformer)
        if transformed[0] != transformed[-1]:
            transformed.append(transformed[0])
        polygons.append([transformed])
    return polygons


def build_geojson() -> Path:
    """Đọc shapefile, lọc 50 frozen zone, ghi GeoJSON WGS84."""
    data_config = load_data_config()
    frozen_zone_ids = set(load_frozen_zone_ids(data_config))
    transformer = _build_transformer()

    reader = shapefile.Reader(str(SHAPEFILE_PATH))
    field_names = [field[0] for field in reader.fields[1:]]
    features = []
    seen_zone_ids = set()

    for record, shape in zip(reader.records(), reader.shapes()):
        attributes = dict(zip(field_names, record))
        zone_id = int(attributes["LocationID"])
        if zone_id not in frozen_zone_ids:
            continue
        polygons = _shape_to_polygons(shape, transformer)
        if not polygons:
            continue
        seen_zone_ids.add(zone_id)
        features.append({
            "type": "Feature",
            "id": zone_id,
            "properties": {
                "pu_location_id": zone_id,
                "zone_name": str(attributes["zone"]).strip(),
                "borough": str(attributes["borough"]).strip(),
            },
            "geometry": {"type": "MultiPolygon", "coordinates": polygons},
        })
    reader.close()

    missing = sorted(frozen_zone_ids - seen_zone_ids)
    if missing:
        raise ValueError(f"Shapefile thiếu geometry cho frozen zone: {missing}")

    features.sort(key=lambda feature: feature["id"])
    geojson = {"type": "FeatureCollection", "features": features}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as stream:
        json.dump(geojson, stream, ensure_ascii=False, separators=(",", ":"))

    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"Đã ghi {len(features)} zone vào {OUTPUT_PATH} ({size_mb:.2f} MB)")
    return OUTPUT_PATH


if __name__ == "__main__":
    build_geojson()
