"""
YOLO Trainer — Dataset Manager
Handles dataset ZIP upload, validation, extraction, listing, and deletion.
Supports YOLO-format datasets with images/, labels/, and data.yaml.
"""

import os
import zipfile
import shutil
from typing import Optional

from config import DATASETS_DIR
from utils import ensure_dir, read_yaml, count_images, count_labels


def validate_zip_structure(zip_path: str) -> dict:
    """Validate that a ZIP file contains the required YOLO dataset structure.

    Returns a dict with:
        valid: bool
        errors: list[str]
        stats: dict { image_count, class_count, class_names, train_images, val_images }
    """
    result = {
        "valid": True,
        "errors": [],
        "stats": {
            "image_count": 0,
            "class_count": 0,
            "class_names": [],
            "train_images": 0,
            "val_images": 0,
        },
    }

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()

            # Strip top-level directory prefix if present (e.g., "dataset/images/..." -> "images/...")
            prefix = _find_common_prefix(namelist)

            has_images = False
            has_labels = False
            has_data_yaml = False
            data_yaml_path = None

            for entry in namelist:
                # Strip prefix for comparison
                relative = _strip_prefix(entry, prefix)
                if relative is None:
                    continue

                if relative == "data.yaml" and not entry.endswith("/"):
                    has_data_yaml = True
                    data_yaml_path = entry

                if relative.startswith("images/") and not entry.endswith("/"):
                    has_images = True

                if relative.startswith("labels/") and not entry.endswith("/"):
                    has_labels = True

            if not has_data_yaml:
                result["errors"].append("缺少 data.yaml 文件（数据集配置文件）")
            if not has_images:
                result["errors"].append("缺少 images/ 目录（图片文件夹）")
            if not has_labels:
                result["errors"].append("缺少 labels/ 目录（标注文件夹）")

            # Try to read and validate data.yaml
            if has_data_yaml and data_yaml_path:
                try:
                    yaml_content = zf.read(data_yaml_path)
                    import yaml
                    data_yaml = yaml.safe_load(yaml_content)
                    if data_yaml:
                        nc = data_yaml.get("nc", 0)
                        names = data_yaml.get("names", [])
                        result["stats"]["class_count"] = nc
                        if isinstance(names, dict):
                            result["stats"]["class_names"] = list(names.values())
                        elif isinstance(names, list):
                            result["stats"]["class_names"] = names
                except Exception as e:
                    result["errors"].append(f"data.yaml 解析失败: {str(e)}")

            # Count images and labels
            train_img = sum(1 for e in namelist
                           if _strip_prefix(e, prefix) and _strip_prefix(e, prefix).startswith("images/train/")
                           and not e.endswith("/"))
            val_img = sum(1 for e in namelist
                         if _strip_prefix(e, prefix) and _strip_prefix(e, prefix).startswith("images/val/")
                         and not e.endswith("/"))
            result["stats"]["train_images"] = train_img
            result["stats"]["val_images"] = val_img
            result["stats"]["image_count"] = train_img + val_img

            if result["stats"]["image_count"] == 0:
                result["errors"].append("未找到任何图片文件")

    except zipfile.BadZipFile:
        result["errors"].append("文件不是有效的 ZIP 压缩包")
    except Exception as e:
        result["errors"].append(f"读取ZIP文件失败: {str(e)}")

    result["valid"] = len(result["errors"]) == 0
    return result


def validate_yolo_annotations(label_dir: str, class_count: int) -> list[str]:
    """Validate all YOLO label files in a directory. Returns list of error messages."""
    errors = []
    if not os.path.isdir(label_dir):
        return [f"标注目录不存在: {label_dir}"]

    for root, _, files in os.walk(label_dir):
        for filename in files:
            if not filename.endswith(".txt"):
                continue
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 5:
                            errors.append(f"{filename}:{line_num} — 格式错误（需要5个值: class cx cy w h）")
                            continue
                        try:
                            cls_id = int(parts[0])
                            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        except ValueError:
                            errors.append(f"{filename}:{line_num} — 数值解析失败")
                            continue

                        if class_count > 0 and cls_id >= class_count:
                            errors.append(
                                f"{filename}:{line_num} — 类别ID {cls_id} 超出范围（共{class_count}类）"
                            )
                        for coord_name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
                            if val < 0 or val > 1:
                                errors.append(
                                    f"{filename}:{line_num} — {coord_name}={val} 超出归一化范围 [0, 1]"
                                )
            except Exception as e:
                errors.append(f"{filename} — 读取失败: {str(e)}")

    return errors


def extract_dataset(zip_path: str, dataset_name: str) -> str:
    """Extract a dataset ZIP to datasets/<dataset_name>/. Returns path to data.yaml."""
    dest_dir = ensure_dir(os.path.join(DATASETS_DIR, dataset_name))

    with zipfile.ZipFile(zip_path, "r") as zf:
        prefix = _find_common_prefix(zf.namelist())
        for member in zf.namelist():
            relative = _strip_prefix(member, prefix)
            if relative is None:
                continue
            target_path = os.path.join(dest_dir, relative)
            if member.endswith("/"):
                os.makedirs(target_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with zf.open(member) as src, open(target_path, "wb") as dst:
                    dst.write(src.read())

    # Fix data.yaml: ensure path is set to current extraction directory
    data_yaml_path = os.path.join(dest_dir, "data.yaml")
    if os.path.exists(data_yaml_path):
        data_yaml = read_yaml(data_yaml_path)
        data_yaml["path"] = dest_dir.replace("\\", "/")
        import yaml
        with open(data_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    return data_yaml_path


def list_datasets() -> list[str]:
    """List names of all available datasets in the datasets directory."""
    if not os.path.isdir(DATASETS_DIR):
        return []
    datasets = []
    for entry in os.scandir(DATASETS_DIR):
        if entry.is_dir():
            data_yaml = os.path.join(entry.path, "data.yaml")
            if os.path.isfile(data_yaml):
                datasets.append(entry.name)
    datasets.sort(key=lambda x: os.path.getmtime(os.path.join(DATASETS_DIR, x)), reverse=True)
    return datasets


def get_dataset_info(dataset_name: str) -> dict:
    """Get detailed information about a stored dataset."""
    ds_dir = os.path.join(DATASETS_DIR, dataset_name)
    data_yaml_path = os.path.join(ds_dir, "data.yaml")

    if not os.path.isfile(data_yaml_path):
        return {"error": f"数据集 '{dataset_name}' 不存在或 data.yaml 缺失"}

    data_yaml = read_yaml(data_yaml_path)

    info = {
        "name": dataset_name,
        "path": ds_dir,
        "data_yaml": data_yaml_path,
        "class_count": data_yaml.get("nc", 0),
        "class_names": [],
        "train_images": 0,
        "val_images": 0,
        "total_images": 0,
        "size_mb": 0.0,
    }

    names = data_yaml.get("names", [])
    if isinstance(names, dict):
        info["class_names"] = list(names.values())
    elif isinstance(names, list):
        info["class_names"] = names

    # Count images
    train_path = os.path.join(ds_dir, "images", "train")
    val_path = os.path.join(ds_dir, "images", "val")
    info["train_images"] = count_images(train_path)
    info["val_images"] = count_images(val_path)
    info["total_images"] = info["train_images"] + info["val_images"]

    # Calculate size
    from utils import get_dir_size_mb
    info["size_mb"] = get_dir_size_mb(ds_dir)

    return info


def delete_dataset(dataset_name: str) -> bool:
    """Delete a dataset directory. Returns True on success."""
    ds_dir = os.path.join(DATASETS_DIR, dataset_name)
    if os.path.isdir(ds_dir):
        shutil.rmtree(ds_dir)
        return True
    return False


def get_dataset_data_yaml_path(dataset_name: str) -> Optional[str]:
    """Get the path to a dataset's data.yaml file."""
    path = os.path.join(DATASETS_DIR, dataset_name, "data.yaml")
    if os.path.isfile(path):
        return path
    return None


# ---- Internal helpers ----

def _find_common_prefix(namelist: list[str]) -> str:
    """Find the top-level directory prefix in a ZIP, if all files share one."""
    top_dirs = set()
    for name in namelist:
        parts = name.split("/")
        if len(parts) > 1 and parts[0]:
            top_dirs.add(parts[0] + "/")
        elif len(parts) == 1 and parts[0]:
            # Files at root level — no common prefix
            return ""
    if len(top_dirs) == 1:
        return list(top_dirs)[0]
    return ""


def _strip_prefix(name: str, prefix: str) -> Optional[str]:
    """Remove the common prefix from a ZIP entry name. Returns None if entry IS the prefix dir."""
    if prefix and name.startswith(prefix):
        relative = name[len(prefix):]
        return relative if relative else None
    return name if name else None
