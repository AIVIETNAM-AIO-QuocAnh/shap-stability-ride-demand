"""Download and validate the twelve official 2025 HVFHV Parquet files."""

from pathlib import Path
import logging
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import pyarrow.parquet as pq
from tqdm import tqdm

from src.configuration import load_data_config


MAX_ATTEMPTS = 3
DOWNLOAD_TIMEOUT_SECONDS = 120
CHUNK_SIZE_BYTES = 8 * 1024 * 1024

LOGGER = logging.getLogger(__name__)


def download_file(url: str, destination: Path) -> Path:
    """Download one file without replacing an existing destination."""
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {destination}")

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
                    f"Downloaded {bytes_written} bytes but the server declared {expected_size} bytes"
                )

            partial_path.replace(destination)
            return destination
        except (HTTPError, URLError, OSError, ValueError) as exc:
            if partial_path.exists():
                partial_path.unlink()
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Could not download {url} after {MAX_ATTEMPTS} attempts"
                ) from exc
            LOGGER.warning(
                "download_attempt_failed",
                extra={"url": url, "attempt": attempt, "error": str(exc)},
            )
            time.sleep(2 ** (attempt - 1))

    raise RuntimeError(f"Download loop ended unexpectedly for {url}")


def validate_parquet_schema(path: Path, required_columns: tuple[str, str]) -> None:
    """Require the raw columns used by the Data pipeline."""
    columns = set(pq.read_schema(path).names)
    missing = set(required_columns) - columns
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def main() -> None:
    """Download and validate every monthly 2025 HVFHV file."""
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
