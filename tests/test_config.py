import os
import json
import pytest
from unittest.mock import patch
from src.core.config import load_config, save_config, get_system_font_options, resource_path


# --- load_config ---

def test_load_config_returns_empty_dict_when_file_missing(tmp_path):
    missing = str(tmp_path / "no_config.json")
    with patch("src.core.config.CONFIG_FILE", missing):
        assert load_config() == {}


def test_load_config_returns_dict_when_file_exists(tmp_path):
    config_path = tmp_path / "config.json"
    data = {"top_n": 20, "font_path": "/test/font.ttf"}
    config_path.write_text(json.dumps(data), encoding="utf-8")
    with patch("src.core.config.CONFIG_FILE", str(config_path)):
        assert load_config() == data


def test_load_config_returns_empty_dict_on_corrupted_json(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("not valid json {{", encoding="utf-8")
    with patch("src.core.config.CONFIG_FILE", str(config_path)):
        assert load_config() == {}


def test_load_config_returns_empty_dict_on_ioerror(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    with patch("src.core.config.CONFIG_FILE", str(config_path)), \
         patch("builtins.open", side_effect=IOError("permission denied")):
        assert load_config() == {}


# --- save_config ---

def test_save_config_creates_file(tmp_path):
    config_path = tmp_path / "config.json"
    with patch("src.core.config.CONFIG_FILE", str(config_path)):
        save_config({"key": "value"})
    assert config_path.exists()


def test_save_config_roundtrip(tmp_path):
    config_path = tmp_path / "config.json"
    data = {"top_n": 15, "stop_words": ["こと", "もの"]}
    with patch("src.core.config.CONFIG_FILE", str(config_path)):
        save_config(data)
        result = load_config()
    assert result == data


def test_save_config_preserves_japanese(tmp_path):
    config_path = tmp_path / "config.json"
    data = {"stop_words": ["こと", "もの", "あれ"]}
    with patch("src.core.config.CONFIG_FILE", str(config_path)):
        save_config(data)
        result = load_config()
    assert result["stop_words"] == ["こと", "もの", "あれ"]


def test_save_config_does_not_raise_on_ioerror(tmp_path):
    config_path = tmp_path / "config.json"
    with patch("src.core.config.CONFIG_FILE", str(config_path)), \
         patch("builtins.open", side_effect=IOError("disk full")):
        # 例外が外に漏れないこと
        save_config({"key": "value"})


# --- get_system_font_options ---

def test_get_system_font_options_returns_dict():
    result = get_system_font_options()
    assert isinstance(result, dict)


def test_get_system_font_options_only_includes_existing_paths():
    result = get_system_font_options()
    for name, path in result.items():
        assert os.path.exists(path), f"存在しないフォントパスが含まれています: {path} ({name})"


def test_get_system_font_options_empty_when_no_fonts_exist():
    with patch("os.path.exists", return_value=False):
        assert get_system_font_options() == {}


def test_get_system_font_options_non_empty_when_all_fonts_exist():
    with patch("os.path.exists", return_value=True):
        result = get_system_font_options()
    assert len(result) > 0


# --- resource_path ---

def test_resource_path_returns_absolute_path():
    result = resource_path("assets/test.txt")
    assert os.path.isabs(result)


def test_resource_path_includes_relative_component():
    result = resource_path("assets/font.ttf")
    assert "assets" in result and "font.ttf" in result
