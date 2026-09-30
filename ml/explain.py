"""NLP extension: turn the two model predictions into a short Thai explanation.

Data-to-text generation: measure colour features of the fish flesh in the image,
compare them with each class's average from the train split, then fill sentence
templates. The text describes what the image looks like relative to typical
Salmon/Trout; it is not a readout of what the CNNs attended to.

    python ml/explain.py --build-stats                        # writes ml/feature_stats.json
    python ml/explain.py <image> --predictions '<json>'      # one JSON line: {summary, features}
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from config import CLASSES, ML_DIR, REPO_DIR

STATS_PATH = ML_DIR / "feature_stats.json"
MIN_FLESH_PIXELS = 200
PRIMARY_MODEL = "densenet"  # higher test accuracy, so it breaks ties when models disagree

# Only features that differ between the classes on the train split (~0.8-1.2 SD apart).
FEATURES = {
    "brightness": {"name": "ความสว่าง", "high": "เนื้อปลาสว่างกว่า", "low": "เนื้อปลาเข้มกว่า"},
    "redness": {"name": "ความแดง", "high": "เนื้อปลาออกโทนแดงมากกว่า", "low": "เนื้อปลาออกโทนส้มมากกว่า"},
    "saturation": {"name": "ความสดของสี", "high": "สีเนื้อปลาสดกว่า", "low": "สีเนื้อปลาซีดกว่า"},
}
MODEL_NAMES = {"densenet": "DenseNet121", "mobilenet": "MobileNetV2"}


def extract_features(image_bgr):
    """Colour stats over orange/red flesh pixels, or None if too little flesh is visible."""
    img = cv2.resize(image_bgr, (224, 224))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    flesh = ((h <= 20) | (h >= 170)) & (s > 60) & (v > 90)
    if flesh.sum() < MIN_FLESH_PIXELS:
        return None
    return {
        "brightness": float(lab[..., 0][flesh].mean() / 2.55),  # 0-100
        "redness": float(lab[..., 1][flesh].mean()) - 128,      # LAB a*
        "saturation": float(s[flesh].mean() / 2.55),            # 0-100
    }


def build_stats():
    from data import load_splits
    per_class = {c: [] for c in CLASSES}
    for path, label in load_splits()["train"]:
        img = cv2.imread(str(REPO_DIR / path))
        f = extract_features(img) if img is not None else None
        if f:
            per_class[CLASSES[label]].append(f)
    stats = {k: {c: {"mean": float(np.mean([f[k] for f in rows])), "std": float(np.std([f[k] for f in rows]))}
                 for c, rows in per_class.items()} for k in FEATURES}
    STATS_PATH.write_text(json.dumps(stats, indent=2))
    return stats


def confidence_word(pct):
    return "สูงมาก" if pct >= 90 else "ค่อนข้างสูง" if pct >= 75 else "ปานกลาง" if pct >= 60 else "ต่ำ"


def degree_word(d):
    return "อย่างชัดเจน" if d >= 1.5 else "พอสมควร" if d >= 0.75 else "เล็กน้อย"


def describe(key, value, stats):
    """Which class this feature value sits closer to, and a sentence saying so."""
    s, t = stats[key]["Salmon"], stats[key]["Trout"]
    mid = (s["mean"] + t["mean"]) / 2
    d = (value - mid) / ((s["std"] + t["std"]) / 2)
    if abs(d) < 0.25:
        return {"feature": key, "value": value, "leans": None, "text": None}
    high = value > mid
    leans = "Salmon" if (s["mean"] > t["mean"]) == high else "Trout"
    text = (f"{FEATURES[key]['high' if high else 'low']}ค่ากลาง{degree_word(abs(d))} "
            f"({FEATURES[key]['name']} {value:.0f} เทียบกับค่าเฉลี่ย Salmon {s['mean']:.0f} / Trout {t['mean']:.0f})")
    return {"feature": key, "value": value, "leans": leans, "text": text}


def explain(image_path, predictions, stats):
    ok = {k: p for k, p in predictions.items() if isinstance(p, dict) and p.get("class")}
    if not ok:
        return {"summary": "ไม่มีผลการทำนายจากโมเดล จึงสร้างคำอธิบายไม่ได้", "features": []}

    names = {k: MODEL_NAMES.get(k, k) for k in ok}
    classes = {p["class"] for p in ok.values()}
    final = ok[PRIMARY_MODEL]["class"] if PRIMARY_MODEL in ok else next(iter(ok.values()))["class"]
    parts = []

    if len(classes) == 1:
        detail = ", ".join(f"{names[k]} มั่นใจ {p['confidence']:.0f}%" for k, p in ok.items())
        who = "ทั้งสองโมเดล" if len(ok) > 1 else names[next(iter(ok))]
        parts.append(f"{who}ทายว่าเป็น {final} ({detail})")
    else:
        detail = ", ".join(f"{names[k]} ทาย {p['class']} ({p['confidence']:.0f}%)" for k, p in ok.items())
        parts.append(f"สองโมเดลเห็นไม่ตรงกัน: {detail} จึงยึดผลของ {MODEL_NAMES[PRIMARY_MODEL]} "
                     f"ซึ่งแม่นกว่าบน test set ได้คำตอบเป็น {final}")

    img = cv2.imread(str(image_path))
    feats = extract_features(img) if img is not None else None
    described = []
    if feats is None:
        parts.append("ตรวจไม่พบบริเวณเนื้อปลาสีส้ม-แดงที่ชัดพอในภาพ จึงอธิบายจากสีของเนื้อปลาไม่ได้")
    else:
        described = [describe(k, v, stats) for k, v in feats.items()]
        support = [d["text"] for d in described if d["leans"] == final]
        against = [d["text"] for d in described if d["leans"] and d["leans"] != final]
        if support:
            parts.append(f"ลักษณะของภาพที่สอดคล้องกับ {final}: " + " และ".join(support))
        else:
            parts.append(f"สีของเนื้อปลาในภาพไม่ได้บ่งชี้ว่าเป็น {final} อย่างชัดเจน")
        if against:
            other = "Trout" if final == "Salmon" else "Salmon"
            parts.append("อย่างไรก็ตาม " + " และ".join(against) + f" ซึ่งเป็นลักษณะที่พบใน {other} มากกว่า")

    top = max(p["confidence"] for p in ok.values() if p["class"] == final)
    parts.append(f"ระดับความมั่นใจโดยรวม: {confidence_word(top)}")
    if len(classes) > 1 or top < 75:
        parts.append("ผลนี้ยังไม่แน่นอน ควรตรวจสอบด้วยสายตาเพิ่มเติม")

    return {"summary": "\n".join(parts), "features": described}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explain Salmon/Trout predictions in Thai")
    parser.add_argument("image_path", nargs="?")
    parser.add_argument("--predictions", default="{}", help='JSON: {"densenet": {...}, "mobilenet": {...}}')
    parser.add_argument("--build-stats", action="store_true")
    args = parser.parse_args()

    if args.build_stats:
        print(json.dumps(build_stats(), indent=2))
        sys.exit(0)
    try:
        stats = json.loads(STATS_PATH.read_text()) if STATS_PATH.exists() else build_stats()
        print(json.dumps(explain(args.image_path, json.loads(args.predictions), stats), ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": f"Explanation failed: {e}"}))
        sys.exit(1)
