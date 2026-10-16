from __future__ import annotations

from pathlib import Path

import pytest

from flawbench.config import Config, load_config
from flawbench.dataset.io import sha256_file
from flawbench.dataset.registry import SourceRegistry
from flawbench.dataset.schema import SourceRecord
from flawbench.exceptions import DatasetError, RegistryError
from tests.conftest import REGISTER_DEFAULTS, write_fixture


def test_registration_records_real_metadata(
    ds_root: Path, registry: SourceRegistry, source: SourceRecord
) -> None:
    d = source.data
    ref = ds_root / "data" / "reference" / "fixture_a.wav"
    assert source.source_id == f"src-{sha256_file(ref)[:12]}"
    assert d["sha256"] == sha256_file(ref)
    assert d["audio_path"] == "fixture_a.wav"  # relative POSIX, no machine path
    assert (d["sample_rate"], d["channels"], d["frames"]) == (16000, 1, 96000)
    assert d["rights"] == {
        "use_confirmed": True,
        "redistribution": "permitted",
        "cleared_for_public_redistribution": True,
    }
    assert registry.get(source.source_id).data == d
    assert [r.source_id for r in registry.all()] == [source.source_id]


def test_unknown_redistribution_is_not_marked_cleared(
    ds_root: Path, registry: SourceRegistry
) -> None:
    kwargs = {**REGISTER_DEFAULTS, "redistribution": "unknown"}
    rec = registry.register(write_fixture(ds_root, "u.wav", seed=2), **kwargs)
    assert rec.data["rights"]["cleared_for_public_redistribution"] is False


@pytest.mark.parametrize(
    "override, fragment",
    [
        ({"transcript_text": None}, "exactly one of"),
        ({"transcript_text": "  "}, "empty"),
        ({"rights_confirmed": False}, "Rights must be confirmed"),
        ({"license": " "}, "Licence"),
        ({"attribution": ""}, "Attribution"),
        ({"source_type": "scraped"}, "source_type"),
        ({"redistribution": "maybe"}, "redistribution"),
        ({"source_id": "bad id"}, "source_id"),
    ],
)
def test_insufficient_metadata_is_rejected_and_nothing_is_written(
    ds_root: Path,
    registry: SourceRegistry,
    override: dict,
    fragment: str,  # type: ignore[type-arg]
) -> None:
    path = write_fixture(ds_root, "x.wav")
    with pytest.raises(RegistryError, match=fragment):
        registry.register(path, **{**REGISTER_DEFAULTS, **override})
    assert not registry.paths.sources_file.exists()


def test_transcript_from_file_and_both_given(ds_root: Path, registry: SourceRegistry) -> None:
    path = write_fixture(ds_root, "t.wav")
    tf = ds_root / "t.txt"
    tf.write_text("hello transcript\n", encoding="utf-8")
    kw = {k: v for k, v in REGISTER_DEFAULTS.items() if k != "transcript_text"}
    with pytest.raises(RegistryError, match="exactly one"):
        registry.register(path, transcript_text="a", transcript_file=tf, **kw)
    rec = registry.register(path, transcript_file=tf, **kw)
    assert rec.data["transcript_text"] == "hello transcript"
    assert rec.data["transcript_file"] == "t.txt"


def test_duplicates_are_refused(
    ds_root: Path, registry: SourceRegistry, source: SourceRecord
) -> None:
    same_audio = ds_root / "data" / "reference" / "fixture_a.wav"
    with pytest.raises(RegistryError, match="already registered"):
        registry.register(same_audio, **REGISTER_DEFAULTS)
    other = write_fixture(ds_root, "b.wav", seed=5)
    with pytest.raises(RegistryError, match="already registered"):
        registry.register(other, source_id=source.source_id, **REGISTER_DEFAULTS)
    assert len(registry.all()) == 1


def test_invalid_audio_is_rejected_without_a_record(
    ds_root: Path, registry: SourceRegistry
) -> None:
    bad = ds_root / "data" / "reference" / "bad.wav"
    bad.write_bytes(b"not audio" * 50)
    with pytest.raises(RegistryError, match="Audio validation failed"):
        registry.register(bad, **REGISTER_DEFAULTS)
    assert not registry.paths.sources_file.exists()
    with pytest.raises(RegistryError, match="not found"):
        registry.register(ds_root / "missing.wav", **REGISTER_DEFAULTS)


def test_file_outside_reference_dir_needs_copy_and_never_overwrites(
    tmp_path: Path, ds_root: Path, registry: SourceRegistry
) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    ext = write_fixture(ds_root, "tmp.wav", seed=3)
    moved = outside / "ext.wav"
    ext.rename(moved)
    original_bytes = moved.read_bytes()
    with pytest.raises(RegistryError, match="outside the reference directory"):
        registry.register(moved, **REGISTER_DEFAULTS)
    rec = registry.register(moved, copy=True, **REGISTER_DEFAULTS)
    copied = ds_root / "data" / "reference" / "ext.wav"
    assert copied.read_bytes() == original_bytes and moved.read_bytes() == original_bytes
    assert rec.data["audio_path"] == "ext.wav"

    # A different file with the same name must not replace the existing reference file.
    clash_dir = tmp_path / "clash"
    clash_dir.mkdir()
    clash = clash_dir / "ext.wav"
    import soundfile as sf

    from tests.conftest import SYN_RATE, synthetic_signal

    sf.write(clash, synthetic_signal(3.0, 9), SYN_RATE)
    with pytest.raises(RegistryError, match="refusing to overwrite"):
        registry.register(clash, copy=True, **REGISTER_DEFAULTS)
    assert copied.read_bytes() == original_bytes


def test_copy_setting_from_configuration(tmp_path: Path, ds_root: Path) -> None:
    cfg: Config = load_config(project_root=ds_root, env={"FLAWBENCH_COPY_REFERENCE": "true"})
    (tmp_path / "data" / "reference").mkdir(parents=True)
    src = write_fixture(tmp_path, "z.wav")  # outside the configured reference dir
    rec = SourceRegistry(cfg).register(src, **REGISTER_DEFAULTS)
    assert (ds_root / "data" / "reference" / "z.wav").is_file() and rec.data[
        "audio_path"
    ] == "z.wav"


def test_checksum_mismatch_and_missing_file_detected(
    ds_root: Path, registry: SourceRegistry, source: SourceRecord
) -> None:
    ref = ds_root / "data" / "reference" / "fixture_a.wav"
    assert registry.verify(source) == []
    data = ref.read_bytes()
    ref.write_bytes(data[:-2] + b"\x01\x01")
    assert "checksum mismatch" in registry.verify(source)[0]
    with pytest.raises(RegistryError, match="checksum mismatch"):
        registry.load_audio(source)
    ref.unlink()
    assert "missing" in registry.verify(source)[0]


def test_corrupt_registry_is_reported(
    ds_root: Path, registry: SourceRegistry, source: SourceRecord
) -> None:
    with registry.paths.sources_file.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    with pytest.raises(DatasetError, match="corrupt"):
        registry.all()
    with pytest.raises(DatasetError):
        registry.register(write_fixture(ds_root, "n.wav", seed=8), **REGISTER_DEFAULTS)


def test_registration_does_not_modify_the_original(ds_root: Path, registry: SourceRegistry) -> None:
    path = write_fixture(ds_root, "keep.wav", seed=4)
    before = path.read_bytes()
    registry.register(path, **REGISTER_DEFAULTS)
    assert path.read_bytes() == before
