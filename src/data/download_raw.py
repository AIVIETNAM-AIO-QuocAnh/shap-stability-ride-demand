"""Tải xuống và validate mười hai file HVFHV Parquet chính thức của năm 2025."""

from pathlib import Path
import logging
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import pyarrow.parquet as pq
from tqdm import tqdm

from src.load_config import load_data_config


MAX_ATTEMPTS = 3
DOWNLOAD_TIMEOUT_SECONDS = 120
CHUNK_SIZE_BYTES = 8 * 1024 * 1024

LOGGER = logging.getLogger(__name__)


def download_file(url: str, destination: Path) -> Path:
    """Tải xuống một file mà không thay thế destination đã tồn tại."""
    if destination.exists():
        raise FileExistsError(f"Từ chối ghi đè file đã tồn tại: {destination}")

    partial_path = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            request = Request(url, headers={"User-Agent": "shap-stability-ride-demand/0.1"})
            with urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                expected_size_header = response.headers.get("Content-Length")
                expected_size = int(expected_size_header) if expected_size_header else None
                bytes_written = 0
                with (
                    partial_path.open("wb") as stream,
                    tqdm(
                        total=expected_size,
                        desc=destination.name,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as progress,
                ):
                    while chunk := response.read(CHUNK_SIZE_BYTES):
                        stream.write(chunk)
                        bytes_written += len(chunk)
                        progress.update(len(chunk))

            if expected_size is not None and bytes_written != expected_size:
                raise OSError(
                    f"Đã tải {bytes_written} byte nhưng server khai báo {expected_size} byte"
                )

            partial_path.replace(destination)
            return destination
        except (HTTPError, URLError, OSError, ValueError) as exc:
            if partial_path.exists():
                partial_path.unlink()
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Không thể tải {url} sau {MAX_ATTEMPTS} lần thử"
                ) from exc
            LOGGER.warning(
                "download_attempt_failed",
                extra={"url": url, "attempt": attempt, "error": str(exc)},
            )
            time.sleep(2 ** (attempt - 1))

    raise RuntimeError(f"Vòng lặp download kết thúc bất ngờ cho {url}")


def validate_parquet_schema(path: Path, required_columns: tuple[str, str]) -> None:
    """Kiểm tra các raw column được Data pipeline sử dụng."""
    columns = set(pq.read_schema(path).names)
    missing = set(required_columns) - columns
    if missing:
        raise ValueError(f"{path} thiếu column bắt buộc: {sorted(missing)}")


def main() -> None:
    """Tải xuống và validate mọi file HVFHV tháng của năm 2025."""
    config = load_data_config()
    raw_dir = config["paths"]["raw_dir"]
    source = config["source"]
    for month_label in config["months"]:
        month = int(month_label[-2:])
        filename = source["filename_pattern"].format(year=config["year"], month=month)
        destination = raw_dir / filename
        url = f"{source['base_url'].rstrip('/')}/{filename}"
        downloaded_path = download_file(url, destination)
        validate_parquet_schema(downloaded_path, source["required_columns"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    main()
