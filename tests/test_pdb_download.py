import gzip
import io
import os
import shutil
import time
from pathlib import Path

import pytest
import requests
import urllib3

import alphafold3tools.pdb_download as pdb_download
import alphafold3tools.structure_stores as structure_stores
from alphafold3tools.pdb_download import (
    DownloadError,
    DownloadingStructureStore,
    DownloadLimitReachedError,
    RemoteFileNotFoundError,
    default_cache_dir,
    download_mmcif,
    ensure_pdb_seqres,
    resolve_seqres_database_path,
    template_download_dir,
)

TESTFILES = Path(__file__).parent.parent / "testfiles"


class _FakeResponse:
    """Minimal stand-in for a streamed ``requests.Response``."""

    def __init__(self, payload: bytes | io.RawIOBase, status_code: int = 200):
        self.raw = io.BytesIO(payload) if isinstance(payload, bytes) else payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error", response=self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _BrokenRaw(io.RawIOBase):
    """A response body whose reads fail like a dropped connection would."""

    def __init__(self, exc: BaseException):
        self._exc = exc

    def readable(self):
        return True

    def readinto(self, b):
        raise self._exc


class TestDefaultCacheDir:
    def test_honours_xdg_cache_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert default_cache_dir() == tmp_path / "alphafold3tools"

    def test_falls_back_to_home_cache(self, monkeypatch, tmp_path):
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        assert default_cache_dir() == tmp_path / ".cache" / "alphafold3tools"


class TestDownloadGzip:
    def test_decompresses_payload(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            pdb_download.requests,
            "get",
            lambda url, **kwargs: _FakeResponse(gzip.compress(b">101m_A mol:protein length:3\nMVL\n")),
        )
        dest = tmp_path / "sub" / "pdb_seqres.txt"
        pdb_download._download_gzip("https://example.org/pdb_seqres.txt.gz", dest)
        assert dest.read_text() == ">101m_A mol:protein length:3\nMVL\n"

    def test_not_found_leaves_no_partial_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            pdb_download.requests,
            "get",
            lambda url, **kwargs: _FakeResponse(b"", status_code=404),
        )
        with pytest.raises(RemoteFileNotFoundError, match="404"):
            pdb_download._download_gzip("https://example.org/missing.gz", tmp_path / "out.txt")
        assert list(tmp_path.iterdir()) == []

    def test_server_error_is_not_reported_as_not_found(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            pdb_download.requests,
            "get",
            lambda url, **kwargs: _FakeResponse(b"", status_code=503),
        )
        with pytest.raises(DownloadError, match="503") as excinfo:
            pdb_download._download_gzip("https://example.org/x.gz", tmp_path / "out.txt")
        assert not isinstance(excinfo.value, RemoteFileNotFoundError)

    @pytest.mark.parametrize(
        "exc",
        [
            urllib3.exceptions.ProtocolError("Connection broken: IncompleteRead"),
            urllib3.exceptions.ReadTimeoutError(
                urllib3.HTTPSConnectionPool("example.org"),  # never connects
                "https://example.org/x.gz",
                "Read timed out.",
            ),
        ],
    )
    def test_broken_body_is_wrapped_and_cleaned_up(self, monkeypatch, tmp_path, exc):
        monkeypatch.setattr(pdb_download.requests, "get", lambda url, **kwargs: _FakeResponse(_BrokenRaw(exc)))
        with pytest.raises(DownloadError):
            pdb_download._download_gzip("https://example.org/x.gz", tmp_path / "out.txt")
        assert list(tmp_path.iterdir()) == []

    def test_corrupt_deflate_stream_is_wrapped(self, monkeypatch, tmp_path):
        # A valid gzip header followed by an invalid deflate block raises zlib.error.
        payload = gzip.compress(b"")[:10] + b"\xff" * 64
        monkeypatch.setattr(pdb_download.requests, "get", lambda url, **kwargs: _FakeResponse(payload))
        with pytest.raises(DownloadError):
            pdb_download._download_gzip("https://example.org/x.gz", tmp_path / "out.txt")
        assert list(tmp_path.iterdir()) == []

    def test_interrupt_leaves_no_partial_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            pdb_download.requests,
            "get",
            lambda url, **kwargs: _FakeResponse(_BrokenRaw(KeyboardInterrupt())),
        )
        with pytest.raises(KeyboardInterrupt):
            pdb_download._download_gzip("https://example.org/x.gz", tmp_path / "out.txt")
        assert list(tmp_path.iterdir()) == []

    def test_unwritable_destination_is_wrapped(self, monkeypatch, tmp_path):
        def boom(url, **kwargs):
            raise AssertionError("must not connect when the destination is unusable")

        monkeypatch.setattr(pdb_download.requests, "get", boom)
        (tmp_path / "not_a_dir").write_text("")
        with pytest.raises(DownloadError):
            pdb_download._download_gzip("https://example.org/x.gz", tmp_path / "not_a_dir" / "out.txt")

    def test_corrupt_payload_keeps_existing_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            pdb_download.requests,
            "get",
            lambda url, **kwargs: _FakeResponse(b"this is not gzip"),
        )
        dest = tmp_path / "out.txt"
        dest.write_text("OLD")
        with pytest.raises(DownloadError):
            pdb_download._download_gzip("https://example.org/corrupt.gz", dest)
        assert dest.read_text() == "OLD"
        assert list(tmp_path.iterdir()) == [dest]

    def test_connection_error_is_wrapped(self, monkeypatch, tmp_path):
        def raise_connection_error(url, **kwargs):
            raise requests.ConnectionError("network is unreachable")

        monkeypatch.setattr(pdb_download.requests, "get", raise_connection_error)
        with pytest.raises(DownloadError, match="network is unreachable"):
            pdb_download._download_gzip("https://example.org/x.gz", tmp_path / "out.txt")
        assert list(tmp_path.iterdir()) == []


class TestEnsurePdbSeqres:
    @staticmethod
    def _fake_download(calls: list[str], content: str = "NEW"):
        def fake(url, dest):
            calls.append(url)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)

        return fake

    @staticmethod
    def _failing_download(url, dest):
        raise DownloadError("network is unreachable")

    def test_downloads_when_cache_missing(self, monkeypatch, tmp_path):
        calls: list[str] = []
        monkeypatch.setattr(pdb_download, "_download_gzip", self._fake_download(calls))
        path = ensure_pdb_seqres(cache_dir=tmp_path)
        assert path == tmp_path / "pdb_seqres.txt"
        assert path.read_text() == "NEW"
        assert calls == [pdb_download.PDB_SEQRES_URL]

    def test_reuses_fresh_cache(self, monkeypatch, tmp_path):
        (tmp_path / "pdb_seqres.txt").write_text("OLD")
        calls: list[str] = []
        monkeypatch.setattr(pdb_download, "_download_gzip", self._fake_download(calls))
        path = ensure_pdb_seqres(cache_dir=tmp_path)
        assert path.read_text() == "OLD"
        assert calls == []

    def test_redownloads_stale_cache(self, monkeypatch, tmp_path):
        cached = tmp_path / "pdb_seqres.txt"
        cached.write_text("OLD")
        eight_days_ago = time.time() - 8 * 24 * 60 * 60
        os.utime(cached, (eight_days_ago, eight_days_ago))
        calls: list[str] = []
        monkeypatch.setattr(pdb_download, "_download_gzip", self._fake_download(calls))
        path = ensure_pdb_seqres(cache_dir=tmp_path)
        assert path.read_text() == "NEW"
        assert len(calls) == 1

    def test_stale_cache_is_used_when_redownload_fails(self, monkeypatch, tmp_path):
        cached = tmp_path / "pdb_seqres.txt"
        cached.write_text("OLD")
        eight_days_ago = time.time() - 8 * 24 * 60 * 60
        os.utime(cached, (eight_days_ago, eight_days_ago))
        monkeypatch.setattr(pdb_download, "_download_gzip", self._failing_download)
        path = ensure_pdb_seqres(cache_dir=tmp_path)
        assert path.read_text() == "OLD"

    def test_missing_cache_and_failed_download_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pdb_download, "_download_gzip", self._failing_download)
        with pytest.raises(DownloadError):
            ensure_pdb_seqres(cache_dir=tmp_path)


class TestResolveSeqresDatabasePath:
    def test_returns_given_path_without_downloading(self, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("must not download when a path is given")

        monkeypatch.setattr(pdb_download, "ensure_pdb_seqres", boom)
        assert resolve_seqres_database_path("/data/pdb_seqres.txt") == "/data/pdb_seqres.txt"

    def test_downloads_when_omitted(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pdb_download, "ensure_pdb_seqres", lambda: tmp_path / "pdb_seqres.txt")
        assert resolve_seqres_database_path(None) == str(tmp_path / "pdb_seqres.txt")

    def test_failure_mentions_cli_option(self, monkeypatch):
        def failing():
            raise DownloadError("network is unreachable")

        monkeypatch.setattr(pdb_download, "ensure_pdb_seqres", failing)
        with pytest.raises(DownloadError, match="--seqres_database_path"):
            resolve_seqres_database_path(None)


class TestDownloadMmcif:
    def test_falls_back_to_mirror(self, monkeypatch, tmp_path):
        urls: list[str] = []

        def fake(url, dest):
            urls.append(url)
            if "files.wwpdb.org" in url:
                raise DownloadError("503 Server Error")
            dest.write_text("data_1BJP\n")

        monkeypatch.setattr(pdb_download, "_download_gzip", fake)
        path = download_mmcif("1BJP", tmp_path)
        assert path == tmp_path / "1bjp.cif"
        assert path.read_text() == "data_1BJP\n"
        assert urls[0].endswith("/divided/mmCIF/bj/1bjp.cif.gz")
        assert len(urls) == 2

    def test_not_found_on_all_sources(self, monkeypatch, tmp_path):
        def failing(url, dest):
            raise RemoteFileNotFoundError("404 Client Error")

        monkeypatch.setattr(pdb_download, "_download_gzip", failing)
        with pytest.raises(RemoteFileNotFoundError, match="9zzz"):
            download_mmcif("9zzz", tmp_path)

    def test_network_failure_is_not_reported_as_not_found(self, monkeypatch, tmp_path):
        def failing(url, dest):
            if "files.wwpdb.org" in url:
                raise RemoteFileNotFoundError("404 Client Error")
            raise DownloadError("network is unreachable")

        monkeypatch.setattr(pdb_download, "_download_gzip", failing)
        with pytest.raises(DownloadError, match="9zzz") as excinfo:
            download_mmcif("9zzz", tmp_path)
        assert not isinstance(excinfo.value, RemoteFileNotFoundError)


def test_template_download_dir():
    assert template_download_dir(Path("out"), "foo") == Path("out") / "templates" / "foo"


class TestDownloadingStructureStore:
    @pytest.fixture
    def fake_download(self, monkeypatch):
        """Replace network downloads with copies from testfiles/mmcif_files."""
        calls: list[str] = []

        def fake(pdb_id, dest_dir):
            calls.append(pdb_id)
            src = TESTFILES / "mmcif_files" / f"{pdb_id.upper()}.cif"
            if not src.exists():
                raise RemoteFileNotFoundError(f"Failed to download the mmCIF file for {pdb_id}.")
            dest = Path(dest_dir) / f"{pdb_id}.cif"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dest)
            return dest

        monkeypatch.setattr(pdb_download, "download_mmcif", fake)
        return calls

    def test_downloads_only_on_access(self, fake_download, tmp_path):
        store_dir = tmp_path / "templates" / "foo"
        store = DownloadingStructureStore(store_dir)
        assert fake_download == []
        assert not store_dir.exists()

        assert "_entry.id   1BJP" in store.get_mmcif_str("1bjp")
        assert fake_download == ["1bjp"]
        assert store.downloaded == ("1bjp",)

        store.get_mmcif_str("1bjp")
        assert fake_download == ["1bjp"]

    def test_existing_file_is_not_downloaded(self, fake_download, tmp_path):
        shutil.copy(TESTFILES / "mmcif_files" / "1BJP.cif", tmp_path / "1bjp.cif")
        store = DownloadingStructureStore(tmp_path)
        store.get_mmcif_str("1bjp")
        assert fake_download == []
        assert store.downloaded == ()

    def test_failed_download_raises_not_found_and_is_not_retried(self, fake_download, tmp_path):
        store = DownloadingStructureStore(tmp_path)
        for _ in range(2):
            with pytest.raises(structure_stores.NotFoundError):
                store.get_mmcif_str("9zzz")
        assert fake_download == ["9zzz"]

    def test_network_failure_propagates(self, monkeypatch, tmp_path):
        def failing(pdb_id, dest_dir):
            raise DownloadError("network is unreachable")

        monkeypatch.setattr(pdb_download, "download_mmcif", failing)
        store = DownloadingStructureStore(tmp_path)
        with pytest.raises(DownloadError) as excinfo:
            store.get_mmcif_str("1bjp")
        assert not isinstance(excinfo.value, structure_stores.NotFoundError)

    def test_download_limit(self, fake_download, tmp_path):
        store = DownloadingStructureStore(tmp_path, max_downloads=2)
        with pytest.raises(structure_stores.NotFoundError):
            store.get_mmcif_str("9zzz")  # failed attempts count toward the limit
        store.get_mmcif_str("1bjp")
        with pytest.raises(DownloadLimitReachedError):
            store.get_mmcif_str("4zz4")
        assert fake_download == ["9zzz", "1bjp"]
        # Files that are already present can still be read after the limit.
        store.get_mmcif_str("1bjp")

    def test_remove_unused_deletes_only_new_rejected_files(self, fake_download, tmp_path):
        shutil.copy(TESTFILES / "mmcif_files" / "7KYZ.cif", tmp_path / "7kyz.cif")
        store = DownloadingStructureStore(tmp_path)
        store.get_mmcif_str("7kyz")  # pre-existing, not downloaded by this store
        store.get_mmcif_str("1bjp")
        store.get_mmcif_str("4zz4")

        store.remove_unused({"1bjp"})

        assert sorted(p.name for p in tmp_path.iterdir()) == ["1bjp.cif", "7kyz.cif"]
        assert store.downloaded == ("1bjp",)
