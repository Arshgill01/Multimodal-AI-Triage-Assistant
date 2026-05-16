import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_final_dataset import IMAGE_REGISTRY, map_kaggle_images


@pytest.fixture
def seed_random():
    random.seed(42)
    np.random.seed(42)


@pytest.fixture
def image_dirs(tmp_path):
    burns = tmp_path / "burns"
    wounds = tmp_path / "wounds"
    burns.mkdir()
    wounds.mkdir()
    for i in range(3):
        (burns / f"burn_{i}.jpg").touch()
        (wounds / f"wound_{i}.jpg").touch()
    return tmp_path


@pytest.fixture
def df_with_images():
    rows = [
        ["p1", "Severe burn on arm", "images/synthetic_p1.jpg", 2, 1],
        ["p2", "Laceration on leg", "images/synthetic_p2.jpg", 4, 0],
        ["p3", "Closed leg fracture", "images/synthetic_p3.jpg", 3, 0],
        ["p4", "Poison ivy rash", "images/synthetic_p4.jpg", 5, 0],
        ["p5", "Chest pain", "images/synthetic_p5.jpg", 2, 1],
    ]
    columns = [
        "patient_id",
        "chief_complaint",
        "image_path",
        "target_esi",
        "flag_high_risk",
    ]
    return pd.DataFrame(rows, columns=columns)


@pytest.fixture
def df_no_images():
    rows = [
        ["p1", "Severe burn", "None", 2, 1],
        ["p2", "Laceration", "None", 4, 0],
    ]
    columns = [
        "patient_id",
        "chief_complaint",
        "image_path",
        "target_esi",
        "flag_high_risk",
    ]
    return pd.DataFrame(rows, columns=columns)


def test_maps_burn_to_burns_dir(image_dirs, df_with_images, seed_random):
    """Rows with 'burn' in chief_complaint get an image from burns/."""
    result = map_kaggle_images(df_with_images, image_base_dir=str(image_dirs))
    row = result[result["patient_id"] == "p1"].iloc[0]
    assert "burns" in row["image_path"]
    assert row["image_path"] != "None"


def test_maps_laceration_to_wounds_dir(image_dirs, df_with_images, seed_random):
    """Rows with 'laceration' in chief_complaint get an image from wounds/."""
    result = map_kaggle_images(df_with_images, image_base_dir=str(image_dirs))
    row = result[result["patient_id"] == "p2"].iloc[0]
    assert "wounds" in row["image_path"]
    assert row["image_path"] != "None"


def test_maps_fracture_to_wounds_dir(image_dirs, df_with_images, seed_random):
    """Rows with 'fracture' in chief_complaint get an image from wounds/."""
    result = map_kaggle_images(df_with_images, image_base_dir=str(image_dirs))
    row = result[result["patient_id"] == "p3"].iloc[0]
    assert "wounds" in row["image_path"]
    assert row["image_path"] != "None"


def test_rash_without_directory_sets_none(image_dirs, df_with_images, seed_random):
    """Rows with 'rash' in chief_complaint fallback to 'None' because no rashes/ dir exists."""
    result = map_kaggle_images(df_with_images, image_base_dir=str(image_dirs))
    row = result[result["patient_id"] == "p4"].iloc[0]
    assert row["image_path"] == "None"


def test_unmatched_complaint_sets_none(image_dirs, df_with_images, seed_random):
    """Rows whose chief_complaint doesn't match any registry keyword get image_path = 'None'."""
    result = map_kaggle_images(df_with_images, image_base_dir=str(image_dirs))
    row = result[result["patient_id"] == "p5"].iloc[0]
    assert row["image_path"] == "None"


def test_no_image_placeholders_unchanged(image_dirs, df_no_images, seed_random):
    """Rows with image_path already set to 'None' are left untouched."""
    original = df_no_images.copy()
    result = map_kaggle_images(df_no_images, image_base_dir=str(image_dirs))
    pd.testing.assert_frame_equal(result, original)


def test_empty_image_dirs_warns(tmp_path, df_with_images, capsys):
    """When no image directories exist, the function prints a WARNING for each missing dir."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    map_kaggle_images(df_with_images, image_base_dir=str(empty_dir))
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "not found" in captured.out


def test_prints_mapped_count(image_dirs, df_with_images, capsys, seed_random):
    """Output includes a 'Mapped' line with per-directory counts."""
    map_kaggle_images(df_with_images, image_base_dir=str(image_dirs))
    captured = capsys.readouterr()
    assert "Mapped" in captured.out
    assert "images mapped" in captured.out


def test_all_images_mapped_count(image_dirs, df_with_images, capsys, seed_random):
    """Verify exact count: 1 burn + 2 wounds = 3 mapped, 2 unmapped."""
    map_kaggle_images(df_with_images, image_base_dir=str(image_dirs))
    captured = capsys.readouterr()
    assert "Burns images mapped: 1" in captured.out
    assert "Wounds images mapped: 2" in captured.out


def test_registry_is_dict_with_expected_keys():
    """IMAGE_REGISTRY is a dict containing the expected mapping keys."""
    assert isinstance(IMAGE_REGISTRY, dict)
    assert "burn" in IMAGE_REGISTRY
    assert "laceration" in IMAGE_REGISTRY
    assert "fracture" in IMAGE_REGISTRY


def test_registry_extensible(image_dirs, df_with_images, capsys, monkeypatch):
    """Adding a new key to IMAGE_REGISTRY works without code changes."""
    monkeypatch.setitem(IMAGE_REGISTRY, "rash", "rashes")
    map_kaggle_images(df_with_images, image_base_dir=str(image_dirs))
    captured = capsys.readouterr()
    assert "not found" in captured.out


def test_image_paths_exist_in_disk(image_dirs, df_with_images, seed_random):
    """Mapped image_path values point to actual files on disk."""
    result = map_kaggle_images(df_with_images, image_base_dir=str(image_dirs))
    for _, row in result.iterrows():
        if row["image_path"] != "None":
            assert row["image_path"].startswith(str(image_dirs))
            assert Path(row["image_path"]).exists()
