#!/usr/bin/env python3
"""Keep only the three curated MT3-style demos in demo_library/recorded.

The script archives old/intermediate demo files instead of deleting them, then
renames the three selected demos to stable names without version suffixes:

  cube_top_grasp
  cuboid_top_yaw_grasp
  tall_cylinder_side_grasp
"""

import json
import os
import re
import shutil
import time


ROOT = os.path.dirname(os.path.abspath(__file__))
DEMO_LIBRARY = os.path.join(ROOT, "demo_library")
RECORDED_DIR = os.path.join(DEMO_LIBRARY, "recorded")
SCENE_DIR = os.path.join(DEMO_LIBRARY, "scene_packages")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def move_path(src, dst):
    if not os.path.exists(src):
        return
    ensure_dir(os.path.dirname(dst))
    if os.path.exists(dst):
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        else:
            os.remove(dst)
    shutil.move(src, dst)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def json_exists(demo_id):
    return os.path.isfile(os.path.join(RECORDED_DIR, demo_id + ".json"))


def latest_numbered_demo(prefix):
    if not os.path.isdir(RECORDED_DIR):
        return None
    best = None
    best_num = -1
    pat = re.compile(r"^%s_v(\d+)\.json$" % re.escape(prefix))
    for name in os.listdir(RECORDED_DIR):
        m = pat.match(name)
        if not m:
            continue
        num = int(m.group(1))
        if num > best_num:
            best_num = num
            best = name[:-5]
    return best


def choose_demo(candidates, latest_prefix=None):
    for demo_id in candidates:
        if json_exists(demo_id):
            return demo_id
    if latest_prefix:
        demo_id = latest_numbered_demo(latest_prefix)
        if demo_id and json_exists(demo_id):
            return demo_id
    return None


def rename_recorded_prefix(src_id, dst_id, backup_dir):
    for name in list(os.listdir(RECORDED_DIR)):
        if not name.startswith(src_id):
            continue
        src = os.path.join(RECORDED_DIR, name)
        new_name = dst_id + name[len(src_id):]
        dst = os.path.join(RECORDED_DIR, new_name)
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        if os.path.exists(dst):
            move_path(dst, os.path.join(backup_dir, new_name))
        move_path(src, dst)


def patch_demo_json(demo_id, object_category, object_label, language_tags):
    path = os.path.join(RECORDED_DIR, demo_id + ".json")
    data = load_json(path)
    data["id"] = demo_id
    if "demo_name" in data:
        data["demo_name"] = demo_id
    obj = data.setdefault("object_info", {})
    obj["category"] = object_category
    obj["label"] = object_label
    tags = list(dict.fromkeys(list(data.get("language_tags", [])) + language_tags))
    data["language_tags"] = tags
    save_json(path, data)


def choose_scene_dir(source_ids, dst_id):
    candidates = []
    for sid in source_ids:
        candidates.append("demo_" + sid)
    for name in candidates:
        path = os.path.join(SCENE_DIR, name)
        if os.path.isdir(path):
            return name
    return None


def rename_scene_package(src_scene_name, dst_id, backup_dir):
    if not src_scene_name:
        return
    src = os.path.join(SCENE_DIR, src_scene_name)
    dst_name = "demo_" + dst_id
    dst = os.path.join(SCENE_DIR, dst_name)
    if os.path.abspath(src) != os.path.abspath(dst):
        if os.path.isdir(dst):
            move_path(dst, os.path.join(backup_dir, dst_name))
        move_path(src, dst)
    metadata_path = os.path.join(dst, "metadata.json")
    if os.path.isfile(metadata_path):
        try:
            meta = load_json(metadata_path)
            meta["name"] = dst_name
            meta["linked_demo_id"] = dst_id
            meta["demo_id"] = dst_id
            save_json(metadata_path, meta)
        except Exception as exc:
            print("WARN: failed to patch scene metadata for %s: %s" % (dst_name, exc))


def archive_unwanted_recorded(keep_ids, backup_dir):
    keep_prefixes = tuple(keep_ids)
    for name in list(os.listdir(RECORDED_DIR)):
        if name.startswith(keep_prefixes):
            continue
        move_path(os.path.join(RECORDED_DIR, name), os.path.join(backup_dir, name))


def archive_unwanted_demo_scenes(keep_ids, backup_dir):
    if not os.path.isdir(SCENE_DIR):
        return
    keep_scene_names = set("demo_" + demo_id for demo_id in keep_ids)
    for name in list(os.listdir(SCENE_DIR)):
        path = os.path.join(SCENE_DIR, name)
        if not os.path.isdir(path):
            continue
        if not name.startswith("demo_"):
            continue
        if name in keep_scene_names:
            continue
        move_path(path, os.path.join(backup_dir, name))


def main():
    if not os.path.isdir(RECORDED_DIR):
        raise RuntimeError("No recorded demo directory: %s" % RECORDED_DIR)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_root = ensure_dir(os.path.join(DEMO_LIBRARY, "archived_before_three_demo_cleanup_" + stamp))
    recorded_backup = ensure_dir(os.path.join(backup_root, "recorded"))
    scene_backup = ensure_dir(os.path.join(backup_root, "scene_packages"))

    specs = [
        {
            "dst": "cube_top_grasp",
            "src_candidates": ["cube_top_grasp", "cube_top_grasp_v2"],
            "scene_candidates": ["cube_top_grasp", "cube_top_grasp_v2_real", "cube_top_grasp_v2"],
            "category": "cube",
            "label": "green_cube",
            "tags": ["pick up green cube", "top grasp green cube", "抓取绿色方块"],
        },
        {
            "dst": "cuboid_top_yaw_grasp",
            "src_candidates": [
                "cuboid_top_yaw_grasp",
                "cuboid_green_top_yaw_grasp_cartesian_v2",
                "cuboid_green_top_yaw_grasp_cartesian_v1",
            ],
            "scene_candidates": [
                "cuboid_top_yaw_grasp",
                "cuboid_green_top_yaw_grasp_cartesian_v2",
                "cuboid_green_top_yaw_grasp_cartesian_v1",
            ],
            "category": "rectangular_prism",
            "label": "green_rectangular_prism",
            "tags": [
                "pick up the green rectangular prism",
                "top yaw grasp green cuboid",
                "旋转夹爪抓取绿色长方体",
            ],
        },
        {
            "dst": "tall_cylinder_side_grasp",
            "src_candidates": ["tall_cylinder_side_grasp", "tall_cylinder_green_side_grasp_v16"],
            "latest_prefix": "tall_cylinder_green_side_grasp",
            "scene_candidates": ["tall_cylinder_side_grasp", "tall_cylinder_green_side_grasp_v16"],
            "category": "cylinder",
            "label": "green_tall_cylinder",
            "tags": [
                "side grasp green tall cylinder",
                "pick up the green tall cylinder from the side",
                "侧边抓取绿色高圆柱",
            ],
        },
    ]

    keep_ids = []
    for spec in specs:
        dst = spec["dst"]
        src = choose_demo(
            spec["src_candidates"],
            latest_prefix=spec.get("latest_prefix"))
        if not src:
            raise RuntimeError("Cannot find source demo for %s" % dst)
        print("Recorded demo: %s -> %s" % (src, dst))
        rename_recorded_prefix(src, dst, recorded_backup)
        patch_demo_json(dst, spec["category"], spec["label"], spec["tags"])
        keep_ids.append(dst)

        scene_sources = list(spec.get("scene_candidates", []))
        if src not in scene_sources:
            scene_sources.insert(0, src)
        scene_name = choose_scene_dir(scene_sources, dst)
        if scene_name:
            print("Scene package: %s -> demo_%s" % (scene_name, dst))
            rename_scene_package(scene_name, dst, scene_backup)
        else:
            print("WARN: no scene package found for %s" % dst)

    archive_unwanted_recorded(keep_ids, recorded_backup)
    archive_unwanted_demo_scenes(keep_ids, scene_backup)

    print("")
    print("Clean demo library ready.")
    print("Kept recorded demos:")
    for demo_id in keep_ids:
        print("  - %s" % demo_id)
    print("Archived old files under:")
    print("  %s" % backup_root)


if __name__ == "__main__":
    main()
