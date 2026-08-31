#!/usr/bin/env python3
"""
MT3 Demonstration Library

Manages a collection of task demonstrations with two-layer retrieval support:
  Layer 1 — Language retrieval: LLM semantic matching with offline fallback
  Layer 2 — Geometric retrieval: shape feature matching via weighted distance

Each demo contains:
  - Language tags for semantic matching
  - Geometric features (dimensions, shape, color) for geometric matching
  - Grasp pose relative to object (for alignment)
  - Approach/retract directions and gripper parameters
"""
import json
import os
import pickle
import math

try:
    from llm_retriever import LLMSemanticRetriever
except Exception:
    LLMSemanticRetriever = None


class DemoLibrary:
    def __init__(self, library_dir=None, execution_environment=None):
        if library_dir is None:
            library_dir = os.path.join(os.path.dirname(__file__), "demo_library")
        self.library_dir = library_dir
        self.execution_environment = self._normalize_execution_environment(
            execution_environment)
        self.demos = []
        self.template = None
        self.semantic_retriever = LLMSemanticRetriever() if LLMSemanticRetriever else None
        self.include_handwritten = os.environ.get("MT3_INCLUDE_HANDWRITTEN_DEMOS", "0") == "1"
        self._load()

    def _normalize_execution_environment(self, execution_environment):
        if execution_environment is None:
            execution_environment = os.environ.get(
                "MT3_EXECUTION_ENVIRONMENT", "simulation")
        text = str(execution_environment or "simulation").strip().lower()
        if text in ("real", "robot", "sawyer_real", "physical"):
            return "real"
        if text in ("sim", "simulation", "gazebo", ""):
            return "simulation"
        return text

    def _recorded_dir(self):
        env_dir = os.path.join(
            self.library_dir, self.execution_environment, "recorded")
        legacy_dir = os.path.join(self.library_dir, "recorded")
        if os.path.isdir(env_dir):
            return env_dir
        if self.execution_environment == "simulation":
            return legacy_dir
        return env_dir

    def _auto_recorded_dir(self):
        env_dir = os.path.join(
            self.library_dir, self.execution_environment, "auto_recorded")
        legacy_dir = os.path.join(self.library_dir, "auto_recorded")
        if os.path.isdir(env_dir):
            return env_dir
        if self.execution_environment == "simulation":
            return legacy_dir
        return env_dir

    def _load(self):
        json_path = os.path.join(self.library_dir, "cube_demos.json")
        pkl_path = os.path.join(self.library_dir, "cube_template.pkl")
        recorded_dir = self._recorded_dir()

        # Handwritten demos are kept only as an optional fallback/reference.
        # By default retrieval uses real recorded demonstrations only.
        if self.include_handwritten and os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
            self.demos = data.get("demos", [])
            print(f"[DemoLibrary] Loaded {len(self.demos)} demos from {json_path}")
        elif os.path.exists(json_path):
            print("[DemoLibrary] Skipped handwritten demos from cube_demos.json "
                  "(set MT3_INCLUDE_HANDWRITTEN_DEMOS=1 to include them)")
        else:
            print(f"[DemoLibrary] No demo file at {json_path}")

        # 加载录制好的demo（带速度轨迹的）
        if os.path.isdir(recorded_dir):
            recorded_count = 0
            for fname in sorted(os.listdir(recorded_dir)):
                if fname.endswith(".json"):
                    fpath = os.path.join(recorded_dir, fname)
                    try:
                        with open(fpath, "r") as f:
                            recorded_demo = json.load(f)
                        # 转换为DemoLibrary统一格式
                        demo_entry = self._convert_recorded_demo(
                            recorded_demo, recorded_dir, source_path=fpath)
                        if demo_entry:
                            self.demos.append(demo_entry)
                            recorded_count += 1
                    except Exception as e:
                        print(f"[DemoLibrary] Failed to load {fname}: {e}")
            print(
                f"[DemoLibrary] Loaded {recorded_count} "
                f"{self.execution_environment} recorded demos from {recorded_dir}")
        else:
            print(
                f"[DemoLibrary] No {self.execution_environment} recorded "
                f"demo directory at {recorded_dir}")

        official_count = self._load_official_semantic_demos()
        print(f"[DemoLibrary] Loaded {official_count} official semantic-only demos")

        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                self.template = pickle.load(f)
            print(f"[DemoLibrary] Loaded geometric template from {pkl_path}")
        else:
            print(f"[DemoLibrary] No template file at {pkl_path}")

    def _load_official_semantic_demos(self):
        """Load official MT3 asset demos for language-retrieval testing only."""
        official_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "mt3", "learning_thousand_tasks",
            "assets", "demonstrations"))
        if not os.path.isdir(official_dir):
            return 0

        count = 0
        for name in sorted(os.listdir(official_dir)):
            path = os.path.join(official_dir, name)
            if not os.path.isdir(path):
                continue
            words = name.replace("_", " ")
            color = "blue" if "blue" in name else "grey" if "grey" in name else "unknown"
            obj = "bottle" if "bottle" in name else "shoe" if "shoe" in name else "object"
            self.demos.append({
                "id": f"official_{name}",
                "language_tags": [
                    words,
                    f"pick up {obj}",
                    f"grab {obj}",
                    f"{color} {obj}",
                    "official mt3 demonstration",
                ],
                "object_category": obj,
                "geometric_features": {
                    "shape": obj,
                    "color_name": color,
                },
                "_semantic_only": True,
                "_source_dir": path,
            })
            count += 1
        return count

    def _convert_recorded_demo(self, recorded, recorded_dir, source_path=None):
        """将录制格式的demo转为DemoLibrary统一格式"""
        obj_info = recorded.get("object_info", {})
        bn = recorded.get("bottleneck_pose_base_frame", {}).get("position_m", {})
        bn_ori = recorded.get("bottleneck_pose_base_frame", {}).get("orientation_xyzw", {})
        obj_pos = obj_info.get("position_base", [0.6, 0.0, -0.58])
        obj_size = obj_info.get("size_m", [0.045, 0.045, 0.045])
        obj_category = obj_info.get("category", "unknown")
        obj_label = obj_info.get("label", obj_category)
        demo_shape = self._canonical_shape(obj_category, obj_label, obj_size)
        aspect = self._aspect_ratio(obj_size)
        approach_direction = recorded.get("approach_direction", [0.0, 0.0, -1.0])
        is_top_grasp = (
            abs(float(approach_direction[0])) < 1e-6
            and abs(float(approach_direction[1])) < 1e-6
            and float(approach_direction[2]) < -0.5
        )
        is_pick_place = recorded.get("task_type") == "pick_place"
        grasp_pose_recorded = None
        if is_pick_place and is_top_grasp:
            # Pick-place execution scripts expect grasp_z to be the object
            # contact/top-surface target. The recorded grasp_pose is the
            # Sawyer right_hand TF frame, so using it directly would add the
            # flange offset twice and make the grasp close in the air.
            clearance = float(recorded.get("top_grasp_clearance_m", 0.005))
            grasp_delta = [0.0, 0.0, float(obj_size[2]) + clearance]
            grasp_pos = [
                float(obj_pos[0]),
                float(obj_pos[1]),
                float(obj_pos[2]) + grasp_delta[2],
            ]
            grasp_ori = {
                "x": bn_ori.get("x", -1.0), "y": bn_ori.get("y", 0.0),
                "z": bn_ori.get("z", 0.0), "w": bn_ori.get("w", 0.0),
            }
        elif not is_top_grasp:
            grasp_pose_recorded = (
                recorded.get("side_grasp_pose_base_frame")
                or self._construct_side_grasp_pose(
                    recorded, obj_pos, obj_size, approach_direction, bn_ori)
                or recorded.get("grasp_pose_base_frame", None)
                or self._estimate_recorded_grasp_pose(recorded)
            )
        else:
            grasp_pose_recorded = recorded.get("grasp_pose_base_frame", None)

        if (not (is_pick_place and is_top_grasp)) and grasp_pose_recorded:
            gp = grasp_pose_recorded.get("position_m", {})
            go = grasp_pose_recorded.get("orientation_xyzw", {})
            grasp_pos = [
                float(gp.get("x", obj_pos[0])),
                float(gp.get("y", obj_pos[1])),
                float(gp.get("z", obj_pos[2])),
            ]
            grasp_ori = {
                "x": float(go.get("x", bn_ori.get("x", 1.0))),
                "y": float(go.get("y", bn_ori.get("y", 0.0))),
                "z": float(go.get("z", bn_ori.get("z", 0.0))),
                "w": float(go.get("w", bn_ori.get("w", 0.0))),
            }
            grasp_delta = [
                grasp_pos[0] - obj_pos[0],
                grasp_pos[1] - obj_pos[1],
                grasp_pos[2] - obj_pos[2],
            ]
        elif not (is_pick_place and is_top_grasp):
            # Top-down demos store the bottleneck above the object. For their
            # execution target, use the demonstrated cube top plus a small
            # clearance. Non-top demos should provide or estimate a real grasp
            # pose from the recorded interaction trajectory.
            grasp_delta = [0.0, 0.0, obj_size[2] + 0.005]
            grasp_pos = [
                obj_pos[0] + grasp_delta[0],
                obj_pos[1] + grasp_delta[1],
                obj_pos[2] + grasp_delta[2],
            ]
            grasp_ori = {
                "x": bn_ori.get("x", 1.0), "y": bn_ori.get("y", 0.0),
                "z": bn_ori.get("z", 0.0), "w": bn_ori.get("w", 0.0),
            }

        place_info = recorded.get("place_info", {}) or {}
        place_relative = None
        try:
            place_pos = place_info.get("place_pose_base_frame", {}).get("position")
            if place_pos and len(place_pos) >= 3:
                place_relative = [
                    float(place_pos[0]) - float(obj_pos[0]),
                    float(place_pos[1]) - float(obj_pos[1]),
                    float(place_pos[2]) - float(obj_pos[2]),
                ]
        except Exception:
            place_relative = None

        entry = {
            "id": recorded.get("id", "recorded_demo"),
            "language_tags": self._expand_recorded_language_tags(recorded, obj_category, obj_label),
            "task_type": recorded.get("task_type", "grasp"),
            "task": recorded.get("task", recorded.get("language_description", "")),
            "top_grasp_reference": recorded.get("top_grasp_reference"),
            "top_grasp_mouth_center_calibration": recorded.get(
                "top_grasp_mouth_center_calibration", {}),
            "object_info": obj_info,
            "place_info": place_info,
            "place_relative_to_object": {
                "delta_position_m": place_relative,
            } if place_relative is not None else {},
            "object_category": obj_category,
            "geometric_features": {
                "shape": demo_shape,
                "dimensions_m": obj_size,
                "aspect_ratio": aspect,
                "color_rgb": [0.0, 1.0, 0.0],
                "color_name": obj_info.get("color", "green"),
                "object_label": obj_label,
            },
            "object_pose_base_frame": {
                "position_m": {"x": obj_pos[0], "y": obj_pos[1], "z": obj_pos[2]},
                "orientation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            "bottleneck_pose_base_frame": {
                "position_m": {"x": bn.get("x", grasp_pos[0]),
                               "y": bn.get("y", grasp_pos[1]),
                               "z": bn.get("z", grasp_pos[2])},
                "orientation_xyzw": {
                    "x": bn_ori.get("x", 1.0), "y": bn_ori.get("y", 0.0),
                    "z": bn_ori.get("z", 0.0), "w": bn_ori.get("w", 0.0),
                },
            },
            "grasp_pose_base_frame": {
                "description": recorded.get("language_description", ""),
                "position_m": {"x": grasp_pos[0], "y": grasp_pos[1], "z": grasp_pos[2]},
                "orientation_xyzw": grasp_ori,
            },
            "grasp_relative_to_object": {
                "delta_position_m": grasp_delta,
                "delta_orientation_xyzw": [1.0, 0.0, 0.0, 0.0],
            },
            "approach_direction": approach_direction,
            "retract_direction": recorded.get("retract_direction", [0.0, 0.0, 1.0]),
            "gripper_opening_m": recorded.get("gripper_opening_m", 0.07),
            "grasp_strategy": recorded.get(
                "grasp_strategy", "top_grasp" if is_top_grasp else "side_grasp"),
            # 附加：录制的速度轨迹（用于真实的MT3速度回放）
            "_recorded_trajectory": recorded.get("trajectory", None),
            "_source_file": recorded.get("id", ""),
            "_recorded_json_path": source_path or os.path.join(
                recorded_dir, "%s.json" % recorded.get("id", "recorded_demo")),
            "_execution_environment": self.execution_environment,
        }
        return entry

    def _estimate_recorded_grasp_pose(self, recorded):
        """Estimate a non-top grasp pose from the recorded interaction phase."""
        traj = recorded.get("trajectory", {}) or {}
        poses = traj.get("poses", []) or []
        if not poses:
            return None

        chosen = None
        for pose in poses:
            if pose.get("gripper_next") == 1 or pose.get("gripper_state") == 1:
                chosen = pose
                break
        if chosen is None:
            chosen = poses[min(len(poses) - 1, max(0, int(len(poses) * 0.75)))]

        pos = chosen.get("position", None)
        ori = chosen.get("orientation", None)
        if not pos or not ori or len(pos) < 3 or len(ori) < 4:
            return None
        return {
            "position_m": {
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
            },
            "orientation_xyzw": {
                "x": float(ori[0]),
                "y": float(ori[1]),
                "z": float(ori[2]),
                "w": float(ori[3]),
            },
        }

    def _construct_side_grasp_pose(self, recorded, obj_pos, obj_size,
                                   approach_direction, bn_ori):
        """Build the intended side-grasp contact pose from object geometry."""
        try:
            d = [float(v) for v in approach_direction[:3]]
            norm = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
            if norm < 1e-9:
                return None
            d = [v / norm for v in d]
            sx, sy, sz = [float(v) for v in obj_size[:3]]

            if abs(d[1]) > abs(d[0]):
                lateral_width = sx
                lateral_vec = [-1.0, 0.0, 0.0]
            else:
                lateral_width = sy
                lateral_vec = [0.0, -1.0, 0.0]
            lateral_offset = recorded.get("side_final_lateral_offset", None)
            if lateral_offset is None:
                lateral_offset = max(0.010, min(0.024, lateral_width * 0.40))
            lateral_offset = float(lateral_offset)

            pos = [
                float(obj_pos[0]) - lateral_vec[0] * lateral_offset,
                float(obj_pos[1]) - lateral_vec[1] * lateral_offset,
                float(obj_pos[2]) + sz / 2.0,
            ]
            ori = recorded.get("side_grasp_orientation_xyzw", None)
            if ori is None:
                ori = [
                    bn_ori.get("x", 1.0), bn_ori.get("y", 0.0),
                    bn_ori.get("z", 0.0), bn_ori.get("w", 0.0),
                ]
            return {
                "position_m": {"x": pos[0], "y": pos[1], "z": pos[2]},
                "orientation_xyzw": {
                    "x": float(ori[0] if isinstance(ori, list) else ori.get("x", 1.0)),
                    "y": float(ori[1] if isinstance(ori, list) else ori.get("y", 0.0)),
                    "z": float(ori[2] if isinstance(ori, list) else ori.get("z", 0.0)),
                    "w": float(ori[3] if isinstance(ori, list) else ori.get("w", 0.0)),
                },
            }
        except Exception:
            return None

    def _canonical_shape(self, category, label="", size=None):
        """Normalize object category/label into retrieval-friendly shapes."""
        text = ("%s %s" % (category or "", label or "")).lower()
        if any(k in text for k in ["cuboid", "rectangular", "prism", "长方体", "闀挎柟"]):
            return "rectangular_prism"
        if any(k in text for k in ["cylinder", "圆柱"]):
            return "cylinder"
        if any(k in text for k in ["ellipsoid", "ellipse", "oval", "椭圆"]):
            return "ellipsoid"
        if any(k in text for k in ["sphere", "ball", "球"]):
            return "sphere"
        if size and len(size) >= 3:
            dims = [float(v) for v in size[:3]]
            if max(dims) - min(dims) > 0.015:
                return "rectangular_prism"
            return "cube"
        return "box"

    def _aspect_ratio(self, size):
        if not size or len(size) < 3:
            return [1.0, 1.0, 1.0]
        dims = [max(float(v), 0.001) for v in size[:3]]
        m = max(dims)
        return [v / m for v in dims]

    def _expand_recorded_language_tags(self, recorded, category, label):
        tags = list(recorded.get("language_tags", []))
        desc = recorded.get("language_description", "")
        if desc:
            tags.append(desc)
        text = ("%s %s %s" % (category or "", label or "", " ".join(tags))).lower()
        if any(k in text for k in ["cuboid", "rectangular", "prism", "长方体", "闀挎柟"]):
            tags.extend([
                "cuboid",
                "rectangular prism",
                "green rectangular prism",
                "long rectangular block",
                "yaw grasp",
                "rotated gripper",
                "short-side grasp",
                "抓取长方体",
                "绿色长方体",
            ])
        return tags

    def language_query(self, query_text, top_k=3):
        """
        Layer 1: Language retrieval.
        Prefer LLM-style semantic matching, then fall back to Jaccard similarity.
        Returns list of (demo, score) sorted by descending score.
        """
        semantic_results = self.semantic_language_query(query_text, top_k=top_k)
        if semantic_results:
            return [(demo, score) for demo, score, _meta in semantic_results]

        return self.jaccard_language_query(query_text, top_k=top_k)

    def semantic_language_query(self, query_text, top_k=3):
        """
        LLM semantic retrieval over natural-language instructions.
        Returns list of (demo, score, metadata).
        """
        if self.semantic_retriever is None:
            return []
        return self.semantic_retriever.query(query_text, self.demos, top_k=top_k)

    def jaccard_language_query(self, query_text, top_k=3):
        """
        Legacy keyword retrieval kept as a deterministic fallback.
        Tokenize query and match against demo language_tags using Jaccard similarity.
        """
        query_tokens = self._simple_language_tokens(query_text)
        if not query_tokens:
            return []

        scored = []
        for demo in self.demos:
            demo_tokens = set()
            for tag in demo.get("language_tags", []):
                demo_tokens.update(self._simple_language_tokens(tag))
            if not demo_tokens:
                continue
            intersection = query_tokens & demo_tokens
            union = query_tokens | demo_tokens
            score = len(intersection) / len(union) if union else 0.0
            scored.append((demo, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _simple_language_tokens(self, text):
        """Tokenize English words and common Chinese task terms for fallback matching."""
        import re

        normalized = (text or "").lower()
        tokens = set(re.findall(r"[a-z0-9]+", normalized))
        for term in ["正方体", "方块", "物块", "绿色", "绿", "抓取", "抓", "拿起", "侧面", "上方"]:
            if term in normalized:
                tokens.add(term)
        return tokens

    def geometric_query(self, detected_features, top_k=3, candidates=None):
        """
        Layer 2: Geometric retrieval.
        Compare detected object features against demo library features.
        Uses weighted Euclidean distance on normalized features.

        detected_features should contain:
          - shape: str (e.g., "box")
          - dimensions_m: [x, y, z] in meters
          - aspect_ratio: [rx, ry, rz] or a single float
          - color_rgb: [r, g, b] (optional)
        """
        scored = []
        demo_iter = candidates if candidates is not None else self.demos
        for demo in demo_iter:
            if demo.get("_semantic_only"):
                continue
            df = demo.get("geometric_features", {})
            if not df:
                continue
            score = self._geometric_score(detected_features, df)
            scored.append((demo, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _geometric_score(self, detected, demo):
        """Compute similarity score between detected features and demo features."""
        weights = {
            "shape": 0.20,
            "dimensions": 0.40,
            "aspect_ratio": 0.25,
            "color": 0.15,
        }
        score = 0.0

        # Shape match (exact match = 1.0, else 0.0)
        if "shape" in weights and "shape" in demo:
            w = weights["shape"]
            detected_shape = self._canonical_shape(detected.get("shape", ""))
            demo_shape = self._canonical_shape(demo.get("shape", ""))
            if detected_shape == demo_shape:
                score += w * 1.0
            elif detected_shape == "rectangular_prism" and demo_shape in ("box", "cube"):
                score += w * 0.15
            else:
                score += w * 0.0

        # Dimension match (normalized by demo dimensions)
        if "dimensions_m" in demo and "dimensions_m" in detected:
            w = weights["dimensions"]
            d_dims = detected["dimensions_m"]
            m_dims = demo["dimensions_m"]
            max_dim = max(max(d_dims), max(m_dims), 0.001)
            sq_diff = sum((a - b) ** 2 for a, b in zip(d_dims, m_dims))
            error = math.sqrt(sq_diff) / (max_dim * math.sqrt(3))
            dim_score = max(0.0, 1.0 - error)
            score += w * dim_score

        # Aspect ratio match
        if "aspect_ratio" in demo:
            w = weights["aspect_ratio"]
            d_ar = detected.get("aspect_ratio", None)
            m_ar = demo["aspect_ratio"]
            if d_ar is not None:
                if isinstance(d_ar, (int, float)):
                    d_ar = [d_ar, d_ar, d_ar]
                if isinstance(m_ar, (int, float)):
                    m_ar = [m_ar, m_ar, m_ar]
                sq_diff = sum((a - b) ** 2 for a, b in zip(d_ar, m_ar))
                m_norm_sq = sum(x ** 2 for x in m_ar)
                m_norm = math.sqrt(m_norm_sq) if m_norm_sq > 0 else 0.001
                error = math.sqrt(sq_diff) / m_norm
                ar_score = max(0.0, 1.0 - error)
                score += w * ar_score
            else:
                score += w * 0.5

        # Color match
        if "color_rgb" in demo and "color_rgb" in detected:
            w = weights["color"]
            d_color = detected["color_rgb"]
            m_color = demo["color_rgb"]
            sq_diff = sum((a - b) ** 2 for a, b in zip(d_color, m_color))
            error = math.sqrt(sq_diff) / math.sqrt(3)
            color_score = max(0.0, 1.0 - error)
            score += w * color_score

        return score

    def _task_matches(self, demo, task_type):
        if not task_type:
            return True
        return demo.get("task_type") == task_type

    def _recorded_demo_path(self, demo):
        source_path = demo.get("_recorded_json_path", "")
        if source_path and os.path.exists(source_path):
            return source_path
        did = demo.get("id", "")
        if not did:
            return ""
        path = os.path.join(self._recorded_dir(), "%s.json" % did)
        return path if os.path.exists(path) else ""

    def hierarchical_query(self, query_text, detected_features, task_type=None,
                           language_top_k=8, language_min_score=0.05,
                           language_keep_margin=0.10,
                           return_metadata=False):
        """
        MT3-style two-stage retrieval:
          1. language filters candidate demos within the requested task type;
          2. geometry ranks only those candidates.

        This replaces the older weighted sum.  The language stage chooses the
        micro-skill/task family, while the geometry stage chooses the closest
        recorded object/scene among those candidates.
        """
        if not self.demos:
            raise ValueError("Demo library is empty. Record at least one demo in demo_library/recorded/.")

        executable = [
            d for d in self.demos
            if (not d.get("_semantic_only")) and self._task_matches(d, task_type)
        ]
        if not executable:
            raise ValueError("No executable recorded demos available for task_type=%s." % task_type)

        lang_ranked = []
        for demo, score in self.language_query(query_text, top_k=len(self.demos)):
            if demo.get("_semantic_only") or not self._task_matches(demo, task_type):
                continue
            lang_ranked.append((demo, float(score)))

        if not lang_ranked:
            # Deterministic fallback: if the language layer cannot score, keep
            # the task-type candidate set and let geometry choose.
            lang_ranked = [(demo, 0.0) for demo in executable]

        positive = [(d, s) for d, s in lang_ranked if s >= float(language_min_score)]
        if positive:
            best_lang = max(s for _d, s in positive)
            cutoff = best_lang - float(language_keep_margin)
            strong = [(d, s) for d, s in positive if s >= cutoff]
            language_candidates = [d for d, _s in strong[:int(language_top_k)]]
        else:
            language_candidates = [d for d, _s in lang_ranked[:int(language_top_k)]]

        if not language_candidates:
            language_candidates = executable

        geo_ranked = self.geometric_query(
            detected_features, top_k=len(language_candidates),
            candidates=language_candidates)
        if not geo_ranked:
            raise ValueError("Language candidates were found, but none had geometric features.")

        best_demo, geo_score = geo_ranked[0]
        lang_scores = {d.get("id", ""): s for d, s in lang_ranked}
        metadata = {
            "retrieval_mode": "language_then_geometry",
            "task_type_filter": task_type or "",
            "language_top_k": int(language_top_k),
            "language_min_score": float(language_min_score),
            "language_keep_margin": float(language_keep_margin),
            "language_candidates": [
                {
                    "id": d.get("id", ""),
                    "task_type": d.get("task_type", ""),
                    "language_score": lang_scores.get(d.get("id", ""), 0.0),
                }
                for d in language_candidates
            ],
            "geometric_candidates": [
                {
                    "id": d.get("id", ""),
                    "task_type": d.get("task_type", ""),
                    "language_score": lang_scores.get(d.get("id", ""), 0.0),
                    "geometry_score": float(score),
                }
                for d, score in geo_ranked
            ],
            "selected_demo_id": best_demo.get("id", ""),
            "selected_demo_path": self._recorded_demo_path(best_demo),
            "language_score": lang_scores.get(best_demo.get("id", ""), 0.0),
            "geometry_score": float(geo_score),
        }
        self.last_retrieval_metadata = metadata

        if return_metadata:
            return best_demo, float(geo_score), metadata
        return best_demo, float(geo_score)

    def weighted_query(self, query_text, detected_features, lang_weight=0.3, geo_weight=0.7):
        """
        Legacy weighted language + geometric retrieval.
        Returns best matching demo.
        """
        if not self.demos:
            raise ValueError("Demo library is empty. Record at least one demo in demo_library/recorded/.")

        executable_demos = [d for d in self.demos if not d.get("_semantic_only")]
        if not executable_demos:
            raise ValueError("No executable recorded demos available. Official demos are semantic-only.")

        lang_results = [
            (demo, score) for demo, score in self.language_query(query_text, top_k=len(self.demos))
            if not demo.get("_semantic_only")
        ]
        geo_results = self.geometric_query(detected_features, top_k=len(executable_demos))

        # Build combined scores
        lang_scores = {d["id"]: s for d, s in lang_results}
        geo_scores = {d["id"]: s for d, s in geo_results}

        all_ids = set(lang_scores.keys()) | set(geo_scores.keys())
        if not all_ids:
            raise ValueError("Demo library is empty or no demos could be scored.")

        combined = {}
        for did in all_ids:
            ls = lang_scores.get(did, 0.0)
            gs = geo_scores.get(did, 0.0)
            combined[did] = lang_weight * ls + geo_weight * gs

        # Find best demo
        demo_map = {d["id"]: d for d in self.demos}
        best_id = max(combined, key=combined.get)
        return demo_map[best_id], combined[best_id]

    def full_query(self, query_text, detected_features, lang_weight=0.3, geo_weight=0.7,
                   task_type=None, retrieval_mode="hierarchical", return_metadata=False):
        """
        Default retrieval entry point.

        The project now uses MT3-style hierarchical retrieval by default:
        language candidate filtering followed by geometric ranking.  The old
        0.3/0.7 weighted score remains available with retrieval_mode="weighted".
        """
        if retrieval_mode == "weighted":
            demo, score = self.weighted_query(
                query_text, detected_features,
                lang_weight=lang_weight, geo_weight=geo_weight)
            metadata = {
                "retrieval_mode": "weighted",
                "lang_weight": float(lang_weight),
                "geo_weight": float(geo_weight),
                "selected_demo_id": demo.get("id", ""),
                "selected_demo_path": self._recorded_demo_path(demo),
                "combined_score": float(score),
            }
            self.last_retrieval_metadata = metadata
            if return_metadata:
                return demo, float(score), metadata
            return demo, float(score)

        return self.hierarchical_query(
            query_text, detected_features,
            task_type=task_type,
            return_metadata=return_metadata)

    def get_grasp_pose(self, demo):
        """Extract grasp pose in base frame from a demo entry."""
        gp = demo.get("grasp_pose_base_frame", {})
        pos = gp.get("position_m", {"x": 0.6, "y": 0.0, "z": -0.58})
        ori = gp.get("orientation_xyzw", {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0})
        return {
            "position": [pos["x"], pos["y"], pos["z"]],
            "orientation": [ori["x"], ori["y"], ori["z"], ori["w"]]
        }

    def get_grasp_relative(self, demo):
        """Extract grasp pose relative to object center."""
        rel = demo.get("grasp_relative_to_object", {})
        delta_pos = rel.get("delta_position_m", [0.0, 0.0, 0.025])
        delta_ori = rel.get("delta_orientation_xyzw", [1.0, 0.0, 0.0, 0.0])
        return {
            "delta_position": delta_pos,
            "delta_orientation": delta_ori
        }

    def get_template_points(self):
        """Return the geometric template point cloud if loaded."""
        if self.template:
            return self.template.get("surface_points"), self.template.get("corners_3d")
        return None, None

    def list_demos(self):
        """List all demo IDs."""
        return [d["id"] for d in self.demos]


# ============================================================
# Test entry point
# ============================================================
if __name__ == "__main__":
    import sys

    library = DemoLibrary()
    print(f"\nLoaded demos: {library.list_demos()}")

    if "--test" in sys.argv:
        # Test language retrieval
        print("\n--- Language Retrieval Test ---")
        for query in [
            "pick up the green cube",
            "pick up the blue metal bottle",
            "pick up the grey shoe",
        ]:
            semantic = library.semantic_language_query(query, top_k=2)
            print(f"\nQuery: '{query}'")
            for demo, score, meta in semantic:
                method = meta.get("method", "semantic")
                reason = meta.get("reason", "")
                print(f"  {demo['id']}: score={score:.3f} [{method}] {reason}")

        # Test geometric retrieval
        print("\n--- Geometric Retrieval Test ---")
        detected = {
            "shape": "box",
            "dimensions_m": [0.045, 0.045, 0.045],
            "aspect_ratio": [1.0, 1.0, 1.0],
            "color_rgb": [1.0, 0.0, 0.0]
        }
        results = library.geometric_query(detected, top_k=2)
        for demo, score in results:
            print(f"  {demo['id']}: score={score:.3f}")

        # Test combined retrieval
        print("\n--- Combined Two-Layer Retrieval Test ---")
        try:
            best_demo, score = library.full_query("pick up the red cube", detected)
            print(f"  Best match: {best_demo['id']} (score={score:.3f})")
            print(f"  Grasp pose: {library.get_grasp_pose(best_demo)}")
        except ValueError as exc:
            print(f"  Skipped: {exc}")
