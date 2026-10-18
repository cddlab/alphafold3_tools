"""Download the PDB files needed for template search from the wwPDB.

When ``--seqres_database_path`` / ``--pdb_database_path`` are omitted, template
search falls back to the helpers in this module:

* ``pdb_seqres.txt`` is downloaded once into the user cache directory and
  re-downloaded when the cached copy is older than ``SEQRES_MAX_AGE``.
* mmCIF files are downloaded on demand by ``DownloadingStructureStore`` while the
  template hits are evaluated, so only the candidates that are actually examined
  are fetched.
"""

import datetime
import gzip
import os
import shutil
import tempfile
import threading
import zlib
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import requests
import urllib3
from loguru import logger

import alphafold3tools.structure_stores as structure_stores

PDB_SEQRES_URL: Final[str] = "https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz"
# Tried in order; the RCSB PDB mirror is used when the wwPDB archive is unavailable.
MMCIF_URL_TEMPLATES: Final[tuple[str, ...]] = (
    "https://files.wwpdb.org/pub/pdb/data/structures/divided/mmCIF/{mid}/{pdb_id}.cif.gz",
    "https://files.rcsb.org/download/{pdb_id}.cif.gz",
)
SEQRES_FILENAME: Final[str] = "pdb_seqres.txt"
SEQRES_MAX_AGE: Final[datetime.timedelta] = datetime.timedelta(days=7)
MAX_MMCIF_DOWNLOADS_PER_SEARCH: Final[int] = 20

_TIMEOUT: Final[tuple[float, float]] = (30.0, 300.0)  # (connect, read) in seconds.
_CHUNK_SIZE: Final[int] = 1 << 20
_SEQRES_LOCK = threading.Lock()


class DownloadError(RuntimeError):
    """Raised when a file could not be downloaded."""


class RemoteFileNotFoundError(DownloadError):
    """Raised when the server reports that the requested file does not exist."""


class DownloadLimitReachedError(Exception):
    """Raised when a DownloadingStructureStore has used up its download budget.

    Deliberately not a NotFoundError: callers iterating over template hits should
    stop instead of skipping to the next hit.
    """


def default_cache_dir() -> Path:
    """Returns ``$XDG_CACHE_HOME/alphafold3tools`` (default ``~/.cache/alphafold3tools``)."""
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache_home) if xdg_cache_home else Path.home() / ".cache"
    return base / "alphafold3tools"


def _download_gzip(url: str, dest: Path) -> None:
    """Downloads a gzip-compressed file from ``url`` and writes it decompressed to ``dest``.

    The data is written to a temporary file in the destination directory first, so
    ``dest`` is either left untouched or atomically replaced by a complete file.

    Raises:
        RemoteFileNotFoundError: if the server responds with 404 or 410.
        DownloadError: if the download, decompression or writing fails otherwise.
    """
    tmp_path: Path | None = None
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "wb") as out, requests.get(url, stream=True, timeout=_TIMEOUT) as response:
            response.raise_for_status()
            # Reading response.raw directly raises urllib3 errors (not wrapped by
            # requests) on dropped connections and zlib.error on corrupt data.
            with gzip.GzipFile(fileobj=response.raw) as decompressed:
                shutil.copyfileobj(decompressed, out, _CHUNK_SIZE)
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, dest)
        tmp_path = None
    except requests.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else None
        if status_code in (404, 410):
            raise RemoteFileNotFoundError(f"Failed to download {url}: {e}") from e
        raise DownloadError(f"Failed to download {url}: {e}") from e
    except (requests.RequestException, urllib3.exceptions.HTTPError, zlib.error, OSError, EOFError) as e:
        raise DownloadError(f"Failed to download {url}: {e}") from e
    finally:
        # Also covers KeyboardInterrupt, so no partial file is left behind.
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def ensure_pdb_seqres(cache_dir: Path | None = None) -> Path:
    """Returns the path to a cached pdb_seqres.txt, downloading it if necessary.

    A cached copy younger than ``SEQRES_MAX_AGE`` is reused. An older copy is
    re-downloaded; if that fails, the older copy is used with a warning.

    Args:
        cache_dir: Directory holding the cached file. Defaults to default_cache_dir().

    Raises:
        DownloadError: if there is no cached copy and the download fails.
    """
    seqres_path = (cache_dir or default_cache_dir()) / SEQRES_FILENAME
    with _SEQRES_LOCK:
        if seqres_path.exists():
            downloaded_at = datetime.datetime.fromtimestamp(seqres_path.stat().st_mtime)
            if datetime.datetime.now() - downloaded_at < SEQRES_MAX_AGE:
                logger.success(
                    f"Using the cached pdb_seqres.txt at {seqres_path} (downloaded on {downloaded_at:%Y-%m-%d})."
                )
                return seqres_path
            logger.warning(
                f"The cached pdb_seqres.txt at {seqres_path} (downloaded on {downloaded_at:%Y-%m-%d}) is older "
                f"than {SEQRES_MAX_AGE.days} days. Downloading the latest version from {PDB_SEQRES_URL}."
            )
            try:
                _download_gzip(PDB_SEQRES_URL, seqres_path)
            except DownloadError as e:
                logger.warning(f"{e}. The existing cached pdb_seqres.txt will be used instead.")
                return seqres_path
        else:
            logger.warning(
                f"Downloading pdb_seqres.txt from {PDB_SEQRES_URL} to {seqres_path}. This may take a few minutes."
            )
            _download_gzip(PDB_SEQRES_URL, seqres_path)
    logger.success(f"Downloaded pdb_seqres.txt to {seqres_path}.")
    return seqres_path


def resolve_seqres_database_path(seqres_database_path: str | os.PathLike[str] | None) -> str:
    """Returns ``seqres_database_path``, or a cached download of pdb_seqres.txt if it is None.

    Raises:
        DownloadError: if pdb_seqres.txt is needed but could not be downloaded.
    """
    if seqres_database_path is not None:
        return os.fspath(seqres_database_path)
    logger.warning(
        "--seqres_database_path was not specified. The PDB SEQRES database (pdb_seqres.txt) will be "
        f"obtained from the wwPDB and stored in the user cache directory ({default_cache_dir()})."
    )
    try:
        return str(ensure_pdb_seqres())
    except DownloadError as e:
        raise DownloadError(
            f"{e}. Please check the network connection, or specify a local pdb_seqres.txt with --seqres_database_path."
        ) from e


def download_mmcif(pdb_id: str, dest_dir: str | os.PathLike[str]) -> Path:
    """Downloads the mmCIF file of ``pdb_id`` to ``dest_dir/<pdb_id>.cif`` (lowercase).

    Raises:
        RemoteFileNotFoundError: if no source has the file.
        DownloadError: if the file could not be downloaded for any other reason.
    """
    pdb_id = pdb_id.lower()
    dest = Path(dest_dir) / f"{pdb_id}.cif"
    errors: list[DownloadError] = []
    for url_template in MMCIF_URL_TEMPLATES:
        url = url_template.format(mid=pdb_id[1:3], pdb_id=pdb_id)
        try:
            _download_gzip(url, dest)
        except DownloadError as e:
            errors.append(e)
            continue
        logger.info(f"Downloaded the mmCIF file for {pdb_id} from {url}.")
        return dest
    message = f"Failed to download the mmCIF file for {pdb_id}: {'; '.join(map(str, errors))}"
    if all(isinstance(e, RemoteFileNotFoundError) for e in errors):
        raise RemoteFileNotFoundError(message)
    raise DownloadError(message)


def template_download_dir(output_dir: str | os.PathLike[str], input_name: str) -> Path:
    """Returns the directory that downloaded template mmCIF files are saved to."""
    return Path(output_dir) / "templates" / input_name


class DownloadingStructureStore(structure_stores.StructureStore):
    """A directory-backed StructureStore that downloads missing mmCIF files on demand.

    Files already present in the directory are used as-is. A missing file is
    downloaded from the wwPDB the first time it is requested, so evaluating
    template hits lazily downloads only the candidates that are examined.

    A structure that does not exist on the servers raises NotFoundError (the hit
    is skipped). Any other download failure (e.g. a network outage) is raised as
    DownloadError so that it is not mistaken for "no templates found".
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        max_downloads: int = MAX_MMCIF_DOWNLOADS_PER_SEARCH,
    ):
        """Initialises the instance.

        Args:
          directory: Directory to read mmCIF files from and download them into. It
            is created on the first download.
          max_downloads: Maximum number of download attempts (including files
            that were not found); further missing files raise
            DownloadLimitReachedError.
        """
        super().__init__(directory)
        self._directory = Path(directory)
        self._max_downloads = max_downloads
        self._num_attempts = 0
        self._failed: set[str] = set()
        self._downloaded: list[str] = []

    @property
    def downloaded(self) -> tuple[str, ...]:
        """Targets newly downloaded by this store that have not been removed."""
        return tuple(self._downloaded)

    def get_mmcif_str(self, target_name: str) -> str:
        if not (self._directory / f"{target_name}.cif").exists():
            self._download(target_name)
        return super().get_mmcif_str(target_name)

    def _download(self, target_name: str) -> None:
        if target_name in self._failed:
            raise structure_stores.NotFoundError(f"{target_name=} could not be downloaded")
        if self._num_attempts >= self._max_downloads:
            raise DownloadLimitReachedError(
                f"Reached the maximum number of mmCIF downloads ({self._max_downloads}) for one template search."
            )
        self._num_attempts += 1
        try:
            download_mmcif(target_name, self._directory)
        except RemoteFileNotFoundError as e:
            self._failed.add(target_name)
            logger.warning(str(e))
            raise structure_stores.NotFoundError(f"{target_name=} could not be downloaded") from e
        self._downloaded.append(target_name)

    def remove_unused(self, keep: Iterable[str]) -> None:
        """Deletes the files downloaded by this store whose target is not in ``keep``.

        Files that existed before this store was used are never deleted.
        """
        keep_set = set(keep)
        for target_name in self._downloaded:
            if target_name not in keep_set:
                (self._directory / f"{target_name}.cif").unlink(missing_ok=True)
                logger.info(f"Removed {target_name}.cif, which was not used as a template.")
        self._downloaded = [t for t in self._downloaded if t in keep_set]
