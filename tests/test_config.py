"""Configuration validation, source parsing and path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from ppe_detection import PROJECT_ROOT
from ppe_detection.config import AppConfig, ConfigError, SourceKind, build_config, load_toml, parse_source


def test_defaults_are_valid_and_use_the_webcam():
    config = build_config()
    assert config.source == "0"  # never a sample video by default
    assert config.device == "cuda" and config.gpu_id == 0  # GPU by default; CPU only when asked for
    assert config.model_path == PROJECT_ROOT / "models" / "dfine_n_coco.onnx"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"conf": 0.0}, "--conf"),
        ({"conf": 1.5}, "--conf"),
        ({"width": 0}, "--width"),
        ({"height": -5}, "--width and --height"),
        ({"width": 10000}, "8K"),
        ({"device": "tpu"}, "--device"),
        ({"gpu_id": -1}, "--gpu-id"),
        ({"fps": 0}, "--fps"),
        ({"backend": "gstreamer"}, "--backend"),
        ({"max_frames": 0}, "--max-frames"),
        ({"min_item_overlap": 0.0}, "min_item_overlap"),
        ({"required_ppe": ("gloves",)}, "required_ppe"),
        ({"classes": ("person", " ")}, "empty class"),
        ({"model": Path("ppe.pt")}, "must be an .onnx"),
        ({"output": Path("out.gif")}, "--output must end"),
    ],
)
def test_invalid_settings_are_rejected(overrides, message):
    with pytest.raises(ConfigError, match=message):
        build_config(**overrides)


def test_model_path_resolves_against_project_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = build_config(model=Path("models/dfine_s_coco.onnx"))
    assert config.model_path == PROJECT_ROOT / "models" / "dfine_s_coco.onnx"


def test_camera_index_sources():
    spec = parse_source("2")
    assert spec.kind is SourceKind.CAMERA and spec.camera_index == 2
    with pytest.raises(ConfigError, match="0 or greater"):
        parse_source("-1")
    with pytest.raises(ConfigError, match="empty"):
        parse_source("  ")


def test_file_sources_resolve_from_cwd_then_project(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    image = tmp_path / "photo.JPG"
    image.write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    assert parse_source("clip.mp4").kind is SourceKind.VIDEO
    assert parse_source("photo.JPG").kind is SourceKind.IMAGE
    # Not under the CWD: falls back to the base directory.
    other = tmp_path / "base"
    other.mkdir()
    (other / "deep.avi").write_bytes(b"x")
    assert parse_source("deep.avi", base_dir=other).path == other / "deep.avi"
    with pytest.raises(ConfigError, match="not found"):
        parse_source("missing.mp4")


def test_toml_config_paths_are_relative_to_the_config_file(tmp_path):
    cfg = tmp_path / "site" / "ppe.toml"
    cfg.parent.mkdir()
    cfg.write_text('source = 1\nmodel = "my_models/x.onnx"\nconf = 0.6\nclasses = ["Person"]\n', encoding="utf-8")
    config = load_toml(cfg)
    assert config.source == "1" and config.conf == 0.6 and config.classes == ("Person",)
    assert config.model_path == (tmp_path / "site" / "my_models" / "x.onnx")
    # A --model typed on the command line is relative to the project instead.
    assert build_config(config, model=Path("models/y.onnx")).model_path == PROJECT_ROOT / "models" / "y.onnx"


def test_toml_errors(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_toml(tmp_path / "nope.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("conf = [", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_toml(bad)
    unknown = tmp_path / "unknown.toml"
    unknown.write_text("colour = 'red'\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown setting"):
        load_toml(unknown)


def test_cli_values_override_config_values():
    base = AppConfig(conf=0.7, width=640)
    config = build_config(base, conf=0.4, width=None)
    assert config.conf == 0.4 and config.width == 640


def test_toml_without_model_keeps_the_project_default(tmp_path):
    cfg = tmp_path / "site.toml"
    cfg.write_text("conf = 0.6\n", encoding="utf-8")
    assert build_config(load_toml(cfg)).model_path == PROJECT_ROOT / "models" / "dfine_n_coco.onnx"


def test_toml_output_is_relative_to_the_config_file(tmp_path, monkeypatch):
    cfg = tmp_path / "site" / "ppe.toml"
    cfg.parent.mkdir()
    cfg.write_text('output = "rec/out.mp4"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert load_toml(cfg).output == tmp_path / "site" / "rec" / "out.mp4"


def test_toml_wrong_types_are_config_errors(tmp_path):
    for text, key in [('conf = "0.6"', "conf"), ("width = true", "width"), ('classes = "Person"', "classes"),
                      ("classes = [1, 2]", "classes")]:
        cfg = tmp_path / "t.toml"
        cfg.write_text(text + "\n", encoding="utf-8")
        with pytest.raises(ConfigError, match=key):
            load_toml(cfg)


def test_toml_with_byte_order_mark_is_accepted(tmp_path):
    cfg = tmp_path / "bom.toml"
    cfg.write_bytes(b"\xef\xbb\xbfconf = 0.7\n")  # what older Windows Notepad writes
    assert load_toml(cfg).conf == 0.7


@pytest.mark.parametrize(
    ("overrides", "message"),
    [({"classes": ()}, "--classes is empty"), ({"required_ppe": ()}, "--required-ppe is empty"),
     ({"fourcc": "MJPEG"}, "--fourcc")],
)
def test_empty_or_malformed_lists_are_rejected(overrides, message):
    with pytest.raises(ConfigError, match=message):
        build_config(**overrides)
