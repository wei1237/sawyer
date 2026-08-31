#!/usr/bin/env python3
"""
LLM semantic retriever for the MT3 demo library.

The module upgrades the original keyword/Jaccard language layer with a
semantic matcher. It prefers an OpenAI-compatible chat API when configured,
and falls back to an offline rule-based semantic scorer so the robot pipeline
can still run in Gazebo without network access or an API key.
"""
import json
import os
import re
import urllib.error
import urllib.request


ACTION_SYNONYMS = {
    "pick": {
        "pick", "pickup", "pick up", "grab", "grasp", "lift", "raise",
        "take", "拿", "拿起", "抓", "抓取", "夹", "夹起", "拾取",
    },
    "place": {
        "place", "put", "set", "drop", "move to", "放", "放下", "放到",
    },
    "side_grasp": {
        "side", "lateral", "horizontal", "from side", "侧面", "侧向", "水平",
    },
    "top_grasp": {
        "top", "down", "vertical", "from above", "上方", "向下", "垂直", "顶部",
    },
}

OBJECT_SYNONYMS = {
    "cube": {"cube", "block", "box", "square", "正方体", "方块", "物块", "积木"},
    "cuboid": {
        "cuboid", "rectangular", "rectangular prism", "prism",
        "long block", "rectangular block", "长方体", "长方块",
        "绿色长方体",
    },
}

COLOR_SYNONYMS = {
    "green": {"green", "绿色", "绿", "青绿色"},
    "red": {"red", "红色", "红"},
    "blue": {"blue", "蓝色", "蓝"},
}

# ── 放置方向关键词 (LLM fallback) ──────────────────────────────
PLACE_DIRECTION_KEYWORDS = {
    "left":  ["left",  "左边", "左侧", "左", "左手边"],
    "right": ["right", "右边", "右侧", "右", "右手边"],
    "front": ["front", "前面", "前方", "前", "正前方", "forward"],
    "back":  ["back",  "后面", "后方", "后", "背后", "behind"],
}

# 默认方向偏移量 (object frame, 米)
PLACE_DIRECTION_OFFSETS = {
    "left":  {"dx": 0.0,  "dy": +0.18},
    "right": {"dx": 0.0,  "dy": -0.18},
    "front": {"dx": +0.15, "dy": 0.0},
    "back":  {"dx": -0.15, "dy": 0.0},
}

# 组合方向关键词 → 自定义偏移 (如 "右前方" = right+front)
COMBINED_DIRECTION_KEYWORDS = {
    "右前":  {"dx": +0.10, "dy": -0.12},
    "右后":  {"dx": -0.10, "dy": -0.12},
    "左前":  {"dx": +0.10, "dy": +0.12},
    "左后":  {"dx": -0.10, "dy": +0.12},
    "前右":  {"dx": +0.10, "dy": -0.12},
    "后右":  {"dx": -0.10, "dy": -0.12},
    "前左":  {"dx": +0.10, "dy": +0.12},
    "后左":  {"dx": -0.10, "dy": +0.12},
}


def _normalize(text):
    return (text or "").strip().lower()


def _tokens(text):
    normalized = _normalize(text)
    words = set(re.findall(r"[a-z0-9]+", normalized))
    for term in re.findall(r"[\u4e00-\u9fff]+", normalized):
        words.add(term)
        for synonyms in list(ACTION_SYNONYMS.values()) + list(OBJECT_SYNONYMS.values()) + list(COLOR_SYNONYMS.values()):
            for synonym in synonyms:
                synonym = _normalize(synonym)
                if synonym and synonym in term:
                    words.add(synonym)
    return words


def _contains_any(text, terms):
    normalized = _normalize(text)
    token_set = _tokens(normalized)
    for term in terms:
        term = _normalize(term)
        if not term:
            continue
        if term in token_set or term in normalized:
            return True
    return False


def _demo_text(demo):
    fields = []
    fields.extend(demo.get("language_tags", []))
    fields.append(demo.get("id", ""))
    fields.append(demo.get("task_type", ""))
    fields.append(demo.get("task", ""))
    fields.append(demo.get("object_category", ""))
    place = demo.get("place_info", {}) or {}
    fields.append(place.get("direction", ""))
    grasp = demo.get("grasp_pose_base_frame", {})
    fields.append(grasp.get("description", ""))
    geo = demo.get("geometric_features", {})
    fields.append(geo.get("shape", ""))
    fields.append(geo.get("color_name", ""))
    return " ".join(str(x) for x in fields if x)


class LLMSemanticRetriever:
    def __init__(self, model=None, api_key=None, api_base=None, timeout=20):
        self.model = model or os.environ.get("MT3_LLM_MODEL", "deepseek-v4-flash")
        self.api_key = (
            api_key
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        self.api_base = (
            api_base
            or os.environ.get("DEEPSEEK_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.deepseek.com"
        ).rstrip("/")
        self.timeout = timeout

    def query(self, query_text, demos, top_k=3):
        """Return (demo, score, metadata) sorted by descending semantic score."""
        if not demos:
            return []

        if self.api_key and os.environ.get("MT3_DISABLE_LLM_API", "0") != "1":
            try:
                llm_results = self._query_openai_compatible(query_text, demos, top_k)
                if llm_results:
                    return llm_results
            except Exception as exc:
                print(f"[LLMRetriever] API unavailable, fallback to offline semantic scorer: {exc}")

        return self._offline_semantic_query(query_text, demos, top_k)

    # ── 放置方向解析 ────────────────────────────────────────────

    def resolve_place_direction(self, query_text):
        """从自然语言指令中提取放置方向, 返回结构化结果.

        优先用 LLM API 理解模糊表达 (如 "放到旁边" "挪到那边"),
        API 不可用时 fallback 到关键词匹配.

        Returns:
            dict: {
                "direction": "right" | "left" | "front" | "back",
                "confidence": 0.0-1.0,
                "reason": str,
                "method": "llm_api" | "keyword_fallback",
                "offset_xy": [dx, dy],
            }
        """
        # 优先 LLM API
        if self.api_key and os.environ.get("MT3_DISABLE_LLM_API", "0") != "1":
            try:
                result = self._llm_resolve_direction(query_text)
                if result:
                    return result
            except Exception as exc:
                print(f"[LLMRetriever] direction API unavailable, fallback to keyword: {exc}")

        return self._keyword_resolve_direction(query_text)

    def _llm_resolve_direction(self, query_text):
        """用 LLM API 解析放置方向，支持单方向和自定义 XY 偏移."""
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You extract the intended PLACE TARGET from a robot "
                        "pick-and-place command. The robot works on a tabletop.\n"
                        "The grasped object is at origin (0,0) in its local frame:\n"
                        "  +X = forward (front), -X = backward (back)\n"
                        "  +Y = left, -Y = right\n\n"
                        "Output FORMAT — choose ONE of two modes:\n\n"
                        "Mode 1 — named direction (simple):\n"
                        '  {"mode":"direction","direction":"left|right|front|back",'
                        '"confidence":0.9,"reason":"..."}\n\n'
                        "Mode 2 — custom offset (when user specifies both axes or "
                        "a precise offset, e.g. \"右前方\", \"放到旁边去\", "
                        "\"往左前挪一点\", \"往右后方移\"):\n"
                        '  {"mode":"custom_offset","offset_xy":[dx,dy],'
                        '"confidence":0.85,"reason":"..."}\n'
                        "  dx,dy in METERS. Typical values: 0.10-0.20 for a normal "
                        "tabletop offset.\n\n"
                        "Examples:\n"
                        '- "放到右边" → {"mode":"direction","direction":"right"}\n'
                        '- "放到右前方" → {"mode":"custom_offset","offset_xy":[0.10,-0.12]}\n'
                        '- "往左后挪" → {"mode":"custom_offset","offset_xy":[-0.10,0.12]}\n'
                        '- "放到旁边去" → {"mode":"direction","direction":"right"}\n'
                        '- "放到桌面左侧" → {"mode":"direction","direction":"left"}\n\n'
                        "Return strict JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "query": query_text,
                    }, ensure_ascii=False),
                },
            ],
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body[:300]}")

        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        mode = parsed.get("mode", "direction")

        if mode == "custom_offset" and "offset_xy" in parsed:
            dx, dy = float(parsed["offset_xy"][0]), float(parsed["offset_xy"][1])
            return {
                "mode": "custom_offset",
                "direction": None,
                "confidence": float(parsed.get("confidence", 0.85)),
                "reason": parsed.get("reason", ""),
                "method": "llm_api",
                "offset_xy": [dx, dy],
            }

        direction = parsed.get("direction", "right")
        if direction not in PLACE_DIRECTION_OFFSETS:
            direction = "right"
        offset = PLACE_DIRECTION_OFFSETS[direction]
        return {
            "mode": "direction",
            "direction": direction,
            "confidence": float(parsed.get("confidence", 0.8)),
            "reason": parsed.get("reason", ""),
            "method": "llm_api",
            "offset_xy": [offset["dx"], offset["dy"]],
        }

    def _keyword_resolve_direction(self, query_text):
        """关键词 fallback: 支持单方向和组合方向."""
        text = _normalize(query_text)

        # 先检查组合方向关键词
        for combo, offset in COMBINED_DIRECTION_KEYWORDS.items():
            if combo in text:
                return {
                    "mode": "custom_offset",
                    "direction": None,
                    "confidence": 0.85,
                    "reason": f"combined direction keyword: {combo}",
                    "method": "keyword_fallback",
                    "offset_xy": [offset["dx"], offset["dy"]],
                }

        # 检查单方向关键词
        matched_dirs = []
        for direction, keywords in PLACE_DIRECTION_KEYWORDS.items():
            if any(k in text for k in keywords):
                matched_dirs.append(direction)

        if len(matched_dirs) >= 2:
            # 组合方向：取两个方向的偏移叠加
            dx = sum(PLACE_DIRECTION_OFFSETS[d]["dx"] for d in matched_dirs)
            dy = sum(PLACE_DIRECTION_OFFSETS[d]["dy"] for d in matched_dirs)
            return {
                "mode": "custom_offset",
                "direction": None,
                "confidence": 0.8,
                "reason": f"combined: {'+'.join(matched_dirs)}",
                "method": "keyword_fallback",
                "offset_xy": [dx, dy],
            }

        if matched_dirs:
            direction = matched_dirs[0]
            offset = PLACE_DIRECTION_OFFSETS[direction]
            return {
                "mode": "direction",
                "direction": direction,
                "confidence": 0.9,
                "reason": f"keyword matched: {direction}",
                "method": "keyword_fallback",
                "offset_xy": [offset["dx"], offset["dy"]],
            }

        # 默认右边
        offset = PLACE_DIRECTION_OFFSETS["right"]
        return {
            "mode": "direction",
            "direction": "right",
            "confidence": 0.3,
            "reason": "no direction keyword found, defaulting to right",
            "method": "keyword_fallback",
            "offset_xy": [offset["dx"], offset["dy"]],
        }

    def _query_openai_compatible(self, query_text, demos, top_k):
        candidates = []
        for demo in demos:
            candidates.append({
                "id": demo.get("id", ""),
                "language_tags": demo.get("language_tags", []),
                "object_category": demo.get("object_category", ""),
                "grasp_description": demo.get("grasp_pose_base_frame", {}).get("description", ""),
                "shape": demo.get("geometric_features", {}).get("shape", ""),
                "color": demo.get("geometric_features", {}).get("color_name", ""),
            })

        system_prompt = (
            "You select the best robot demonstration for a natural-language "
            "instruction. Match semantics, not only exact words. Prefer top-down "
            "grasp demos for generic pick/lift/grab instructions unless the user "
            "explicitly asks for side/lateral grasp. Return strict JSON only."
        )
        user_prompt = {
            "query": query_text,
            "candidates": candidates,
            "response_schema": {
                "matches": [
                    {
                        "id": "demo id",
                        "score": "0.0 to 1.0",
                        "canonical_task": "short normalized task",
                        "reason": "brief reason",
                    }
                ]
            }
        }
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body[:300]}")

        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        demo_map = {demo.get("id"): demo for demo in demos}
        results = []
        for item in parsed.get("matches", []):
            demo = demo_map.get(item.get("id"))
            if not demo:
                continue
            score = float(item.get("score", 0.0))
            meta = {
                "method": "llm_api",
                "canonical_task": item.get("canonical_task", ""),
                "reason": item.get("reason", ""),
            }
            results.append((demo, max(0.0, min(1.0, score)), meta))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _offline_semantic_query(self, query_text, demos, top_k):
        results = []
        for demo in demos:
            score, reason, canonical = self._offline_score(query_text, demo)
            results.append((demo, score, {
                "method": "offline_semantic",
                "canonical_task": canonical,
                "reason": reason,
            }))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max(0, top_k)]

    def _offline_score(self, query_text, demo):
        query = _normalize(query_text)
        demo_text = _normalize(_demo_text(demo))
        query_tokens = _tokens(query)
        demo_tokens = _tokens(demo_text)

        score = 0.0
        reasons = []

        if _contains_any(query, ACTION_SYNONYMS["pick"]):
            score += 0.25
            reasons.append("pick/lift intent")
        if _contains_any(query, ACTION_SYNONYMS["place"]):
            if demo.get("task_type") == "pick_place" or "pick and place" in demo_text:
                score += 0.30
                reasons.append("pick-place demo matches place intent")
            else:
                score -= 0.10
                reasons.append("place intent but demo is grasp-only")

        if _contains_any(query, OBJECT_SYNONYMS["cuboid"]) and _contains_any(demo_text, OBJECT_SYNONYMS["cuboid"]):
            score += 0.35
            reasons.append("cuboid/rectangular prism object match")
        elif _contains_any(query, OBJECT_SYNONYMS["cube"]) and _contains_any(demo_text, OBJECT_SYNONYMS["cube"]):
            score += 0.25
            reasons.append("cube/block object match")

        for color, synonyms in COLOR_SYNONYMS.items():
            if _contains_any(query, synonyms) and _contains_any(demo_text, synonyms):
                score += 0.20
                reasons.append(f"{color} color match")

        wants_side = _contains_any(query, ACTION_SYNONYMS["side_grasp"])
        wants_top = _contains_any(query, ACTION_SYNONYMS["top_grasp"])
        demo_id = _normalize(demo.get("id", ""))
        approach = demo.get("approach_direction", [])
        is_side_demo = "side" in demo_id or (approach and abs(approach[2]) < 0.5)
        is_top_demo = "top" in demo_id or (approach and approach[2] < -0.5)

        if wants_side and is_side_demo:
            score += 0.20
            reasons.append("side grasp requested")
        elif wants_side and is_top_demo:
            score -= 0.15
        elif (wants_top or _contains_any(query, ACTION_SYNONYMS["pick"])) and is_top_demo:
            score += 0.15
            reasons.append("top grasp preferred for pick")

        if query_tokens and demo_tokens:
            overlap = len(query_tokens & demo_tokens) / len(query_tokens | demo_tokens)
            score += 0.15 * overlap
            if overlap > 0:
                reasons.append(f"token overlap {overlap:.2f}")

        canonical_parts = []
        if _contains_any(query, ACTION_SYNONYMS["pick"]):
            canonical_parts.append("pick up")
        elif _contains_any(query, ACTION_SYNONYMS["place"]):
            canonical_parts.append("place")

        matched_color = None
        for color, synonyms in COLOR_SYNONYMS.items():
            if _contains_any(query, synonyms) or _contains_any(demo_text, synonyms):
                matched_color = color
                break
        if matched_color:
            canonical_parts.append(matched_color)

        if _contains_any(query, OBJECT_SYNONYMS["cuboid"]) or _contains_any(demo_text, OBJECT_SYNONYMS["cuboid"]):
            canonical_parts.append("rectangular prism")
        elif _contains_any(query, OBJECT_SYNONYMS["cube"]) or _contains_any(demo_text, OBJECT_SYNONYMS["cube"]):
            canonical_parts.append("cube")

        canonical = " ".join(canonical_parts) if canonical_parts else query_text
        return max(0.0, min(1.0, score)), "; ".join(reasons), canonical
