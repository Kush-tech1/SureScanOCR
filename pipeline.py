"""
╔══════════════════════════════════════════════════════════════╗
║          FULL METER READING PIPELINE                         ║
║                                                              ║
║  Stage 1 : Meter Body + Serial Number Detection              ║
║            (RF-DETR ONNX — inference_model_working_sn+mbody) ║
║  Stage 2A: Brand Classification                              ║
║            (EfficientNet-B0 PyTorch — meter_classifier.pth)  ║
║  Stage 2B: Serial Number OCR (runs in parallel with 2A)      ║
║            (CTC ONNX — sn_ocr.onnx)                         ║
║  Stage 3 : Perspective Dewarp                                ║
║            (TorchScript Keypoint — deploy_kp_model.pt)       ║
║  Stage 4 : Digit / Meter Reading                             ║
║            (RF-DETR ONNX — inference_model_digits.onnx)      ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python pipeline.py --image path/to/image.jpg
    python pipeline.py --image path/to/folder/
    python pipeline.py --image img.jpg --output results/ --save_vis
    python pipeline.py --image test --save_crops
"""

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import timm
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms

# ════════════════════════════════════════════════════════════════
# CONFIG  — edit paths & thresholds here
# ════════════════════════════════════════════════════════════════
CFG = {
    # Model paths
    "detection_model":    r"D:\Instinct_4.0_Final_models\rfdetr_stafge1\inference_model_working_sn+mbody.onnx",
    "classifier_model":   r"D:\Instinct_4.0_Final_models\brand_classifier_effientnet_b0\meter_classifier.pth",
    "ocr_model":          r"D:\\Instinct_4.0_Final_models\\paddleocr_sn\\sn_ocr.onnx",
    "ocr_dict":           r"D:\Instinct_4.0_Final_models\paddleocr_sn\dict.txt",
    "dewarp_model":       r"D:\Instinct_4.0_Final_models\keypoint_rcnn\deploy_kp_model.pt",
    "digit_model":        r"D:\Instinct_4.0_Final_models\rfdetr_stage2_digits_dectetor\inference_model_digits.onnx",

    # Thresholds
    "detect_thr":         0.3,   # Stage 1 — lower if body/SN not found
    "dewarp_thr":         0.5,   # Stage 3 — keypoint confidence
    "digit_thr":          0.7,   # Stage 4 — digit detection

    # Image sizes
    "detect_size":        512,
    "digit_size":         512,
    "classifier_size":    224,

    # Stage 1 class names (must match your data.yaml order)
    # Common orderings — check [INFO] debug output on first run to verify
    # e.g. ["mbody", "sn"]  or  ["sn", "mbody"]  or  ["meter", "serial_number"]
    "detect_classes":     ["display_kp", "SN", "meter_body"],

    # Stage 4 digit classes (must match data.yaml order)
    "digit_classes": [
        ".", "0", "1", "2", "3", "4",
        "5", "6", "7", "8", "9",
        "KVah", "Kw", "MD", "colon", "kVA"
    ],

    "default_unit":       "kWh",
    "unit_classes":       {"KVah", "Kw", "MD", "kVA"},
    "punct_classes":      {".", "colon"},

    # Output
    "output_dir":         "pipeline_results",
}
# ════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def preprocess_rfdetr(img_pil, size):
    """Resize, normalise, NCHW float32 — for RF-DETR ONNX models."""
    orig_w, orig_h = img_pil.size
    arr = np.array(img_pil.resize((size, size))).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std
    arr  = np.transpose(arr, (2, 0, 1))[None]
    return orig_w, orig_h, arr


def decode_rfdetr(session, arr, orig_w, orig_h, score_thr):
    """Run RF-DETR session and return (pixel_boxes, scores, class_ids)."""
    out_names = [o.name for o in session.get_outputs()]
    inp_name  = session.get_inputs()[0].name
    outputs   = session.run(None, {inp_name: arr})
    out_map   = dict(zip(out_names, outputs))

    boxes_key  = next(k for k in out_map if out_map[k].shape[-1] == 4)
    logits_key = next(k for k in out_map if k != boxes_key)

    boxes_raw  = out_map[boxes_key][0]
    logits_raw = out_map[logits_key][0]

    probs     = softmax(logits_raw)
    scores    = probs.max(axis=-1)
    class_ids = probs.argmax(axis=-1).astype(int)

    cx, cy = boxes_raw[:, 0], boxes_raw[:, 1]
    bw, bh = boxes_raw[:, 2], boxes_raw[:, 3]
    x1 = (cx - bw / 2) * orig_w
    y1 = (cy - bh / 2) * orig_h
    x2 = (cx + bw / 2) * orig_w
    y2 = (cy + bh / 2) * orig_h
    pixel_boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep = scores >= score_thr
    return pixel_boxes[keep], scores[keep], class_ids[keep]


def crop_pil(img_pil, box, pad_px=4):
    """Crop PIL image from [x1,y1,x2,y2] with optional padding."""
    w, h = img_pil.size
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1) - pad_px)
    y1 = max(0, int(y1) - pad_px)
    x2 = min(w, int(x2) + pad_px)
    y2 = min(h, int(y2) + pad_px)
    return img_pil.crop((x1, y1, x2, y2))


def best_box_for_class(boxes, scores, class_ids, target_id):
    """Return the highest-score box for a given class id, or None."""
    mask = class_ids == target_id
    if not mask.any():
        return None, None
    idx = scores[mask].argmax()
    return boxes[mask][idx], scores[mask][idx]


# ──────────────────────────────────────────────────────────────
# STAGE 1 — Meter Body + Serial Number Detection
# ──────────────────────────────────────────────────────────────
class StageDetection:
    def __init__(self):
        print("[Stage 1] Loading detection model …")
        self.session = ort.InferenceSession(
            CFG["detection_model"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.classes = CFG["detect_classes"]
        # Determine class IDs — support both orderings
        self._meter_id = next(
            (i for i, n in enumerate(self.classes) if n.lower() == "meter_body"), 2
        )
        self._sn_id = next(
            (i for i, n in enumerate(self.classes) if n.upper() == "SN"), 1
        )
        print(f"  [INFO] detect_classes={self.classes}")
        print(f"  [INFO] meter_id={self._meter_id}, sn_id={self._sn_id}")

    def run(self, img_pil):
        orig_w, orig_h, arr = preprocess_rfdetr(img_pil, CFG["detect_size"])
        boxes, scores, class_ids = decode_rfdetr(
            self.session, arr, orig_w, orig_h, CFG["detect_thr"]
        )

        # Debug: print all detected class IDs and scores to verify class ordering
        if len(scores) > 0:
            unique_ids = np.unique(class_ids)
            print(f"  [DEBUG] Raw detections above thr: {len(scores)}")
            for uid in unique_ids:
                mask = class_ids == uid
                top_score = scores[mask].max()
                name = self.classes[uid] if uid < len(self.classes) else f"cls{uid}"
                print(f"  [DEBUG]   class_id={uid} name={name!r}  count={mask.sum()}  top_score={top_score:.3f}")
        else:
            all_probs = softmax(decode_rfdetr.__wrapped__ if hasattr(decode_rfdetr, '__wrapped__') else [])
            print(f"  [DEBUG] No detections above threshold={CFG['detect_thr']}")

        meter_box, meter_score = best_box_for_class(boxes, scores, class_ids, self._meter_id)
        sn_box,    sn_score    = best_box_for_class(boxes, scores, class_ids, self._sn_id)

        meter_crop = crop_pil(img_pil, meter_box) if meter_box is not None else None
        sn_crop    = crop_pil(img_pil, sn_box)    if sn_box    is not None else None

        print(f"  → Meter body : {'found (score={:.3f})'.format(meter_score) if meter_score is not None else 'NOT FOUND'}")
        print(f"  → Serial No  : {'found (score={:.3f})'.format(sn_score)    if sn_score    is not None else 'NOT FOUND'}")

        return {
            "meter_crop":  meter_crop,
            "meter_box":   meter_box,
            "meter_score": float(meter_score) if meter_score is not None else None,
            "sn_crop":     sn_crop,
            "sn_box":      sn_box,
            "sn_score":    float(sn_score)    if sn_score    is not None else None,
            "all_boxes":   boxes,
            "all_scores":  scores,
            "all_class_ids": class_ids,
        }


# ──────────────────────────────────────────────────────────────
# STAGE 2A — Brand Classification
# ──────────────────────────────────────────────────────────────
class StageBrandClassifier:
    def __init__(self):
        print("[Stage 2A] Loading brand classifier …")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(CFG["classifier_model"], map_location=device)
        self.class_names = checkpoint["class_names"]
        self.model = timm.create_model(
            "efficientnet_b0", pretrained=False, num_classes=len(self.class_names)
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device).eval()
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((CFG["classifier_size"], CFG["classifier_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def run(self, meter_crop):
        if meter_crop is None:
            print("  → Brand : SKIPPED (no meter crop)")
            return {"brand": None, "brand_conf": None}

        tensor = self.transform(meter_crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out  = self.model(tensor)
            prob = torch.softmax(out, dim=1)
            conf, idx = torch.max(prob, 1)
        brand = self.class_names[idx.item()]
        score = conf.item()
        print(f"  → Brand : {brand}  (conf={score:.3f})")
        return {"brand": brand, "brand_conf": score}


# ──────────────────────────────────────────────────────────────
# STAGE 2B — Serial Number OCR (CTC)
# ──────────────────────────────────────────────────────────────
class StageSerialOCR:
    def __init__(self):
        print("[Stage 2B] Loading serial OCR model …")
        self.session    = ort.InferenceSession(CFG["ocr_model"])
        self.input_name = self.session.get_inputs()[0].name
        self.chars      = self._load_dict(CFG["ocr_dict"])

    @staticmethod
    def _load_dict(path):
        with open(path, "r", encoding="utf-8") as f:
            chars = [line.strip() for line in f.readlines()]
        return ["blank"] + chars

    def _resize_norm(self, img_pil, shape=(3, 48, 320)):
        imgC, imgH, imgW = shape
        img = np.array(img_pil.convert("RGB"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        resized_w = min(imgW, int(np.ceil(imgH * (w / float(h)))))
        resized   = cv2.resize(img, (resized_w, imgH)).astype("float32")
        resized   = (resized / 255.0 - 0.5) / 0.5
        resized   = resized.transpose((2, 0, 1))
        pad       = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        pad[:, :, :resized_w] = resized
        return pad

    def _ctc_decode(self, preds):
        # preds may be [1, T, C] or [T, C] — normalise to [T, C]
        if preds.ndim == 3:
            preds = preds[0]
        indices = np.argmax(preds, axis=1)   # [T] numpy array
        chars, confs, prev = [], [], -1
        for t in range(len(indices)):
            idx = int(indices[t])            # guaranteed plain Python int
            if idx != 0 and idx != prev:
                chars.append(self.chars[idx])
                confs.append(float(preds[t, idx]))
            prev = idx
        text = re.sub(r"[^A-Z0-9]", "", "".join(chars))
        conf = float(np.mean(confs)) if confs else 0.0
        return text, conf

    def run(self, sn_crop):
        if sn_crop is None:
            print("  → Serial : SKIPPED (no serial crop)")
            return {"serial_number": None, "serial_conf": None}

        arr   = self._resize_norm(sn_crop)[None]
        preds = self.session.run(None, {self.input_name: arr})[0]  # shape: [T, C]
        text, conf = self._ctc_decode(preds)
        print(f"  → Serial : {text}  (conf={conf:.3f})")
        return {"serial_number": text, "serial_conf": conf}


# ──────────────────────────────────────────────────────────────
# STAGE 3 — Perspective Dewarp
# ──────────────────────────────────────────────────────────────
class StageDewarp:
    def __init__(self):
        print("[Stage 3] Loading dewarp keypoint model …")
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(CFG["dewarp_model"], map_location=device)
        self.model.eval()
        self.device = device

    @staticmethod
    def _perspective_warp(image_np, kps):
        tl, tr, br, bl = kps[:, :2]
        width_a  = np.linalg.norm(br - bl)
        width_b  = np.linalg.norm(tr - tl)
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_w = max(int(width_a),  int(width_b))
        max_h = max(int(height_a), int(height_b))
        dst   = np.array([[0,0],[max_w-1,0],[max_w-1,max_h-1],[0,max_h-1]], dtype="float32")
        src   = kps[:, :2].astype("float32")
        M     = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image_np, M, (max_w, max_h))

    def run(self, meter_crop):
        if meter_crop is None:
            print("  → Dewarp : SKIPPED (no meter crop)")
            return {"dewarped_crop": None, "keypoints": None, "dewarp_score": None}

        tensor = F.to_tensor(meter_crop).to(self.device)
        with torch.no_grad():
            _, detections = self.model([tensor])
        output = detections[0]

        # Exactly as in original working inference — take highest scoring detection directly
        scores = output["scores"].cpu().numpy()
        if len(scores) == 0 or scores[0] < CFG["dewarp_thr"]:
            best = scores[0] if len(scores) > 0 else 0.0
            print(f"  → Dewarp : no confident keypoints (best={best:.3f}), using raw crop")
            return {"dewarped_crop": meter_crop, "keypoints": None, "dewarp_score": None}

        # keypoints[0] = best detection, shape [4, 3] -> [x, y, visibility]
        kps        = output["keypoints"][0].cpu().numpy()
        image_np   = np.array(meter_crop)
        warped_np  = self._perspective_warp(image_np, kps)
        warped_pil = Image.fromarray(warped_np)
        print(f"  → Dewarp : OK  (score={scores[0]:.3f})")
        return {"dewarped_crop": warped_pil, "keypoints": kps, "dewarp_score": float(scores[0])}


# ──────────────────────────────────────────────────────────────
# STAGE 4 — Digit / Meter Reading
# ──────────────────────────────────────────────────────────────
class StageDigitReading:
    def __init__(self):
        print("[Stage 4] Loading digit reading model …")
        self.session    = ort.InferenceSession(
            CFG["digit_model"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.classes    = CFG["digit_classes"]
        self.unit_cls   = CFG["unit_classes"]
        self.punct_cls  = CFG["punct_classes"]
        self.def_unit   = CFG["default_unit"]

    def _assemble(self, boxes, scores, class_ids):
        if len(scores) == 0:
            return {"reading": f"N/A {self.def_unit}", "tokens": [],
                    "unit": self.def_unit, "unit_conf": None, "avg_conf": 0.0}

        cx    = (boxes[:, 0] + boxes[:, 2]) / 2.0
        order = np.argsort(cx)
        tokens, unit_char, unit_conf = [], None, None

        for idx in order:
            cid   = int(class_ids[idx])
            name  = self.classes[cid] if cid < len(self.classes) else f"cls{cid}"
            score = float(scores[idx])
            if name in self.unit_cls:
                if unit_conf is None or score > unit_conf:
                    unit_char, unit_conf = name, score
            else:
                tokens.append({
                    "char":     ":" if name == "colon" else name,
                    "raw_name": name,
                    "conf":     score,
                    "box":      boxes[idx],
                    "type":     "punct" if name in self.punct_cls else "digit",
                })

        reading_str  = "".join(t["char"] for t in tokens)
        unit         = unit_char or self.def_unit
        digit_tokens = [t for t in tokens if t["type"] == "digit"]
        avg_conf     = float(np.mean([t["conf"] for t in digit_tokens])) if digit_tokens else 0.0
        return {
            "reading":   f"{reading_str} {unit}",
            "tokens":    tokens,
            "unit":      unit,
            "unit_conf": unit_conf,
            "avg_conf":  avg_conf,
        }

    def run(self, dewarped_crop):
        if dewarped_crop is None:
            print("  → Digits : SKIPPED (no input image)")
            return {"reading": None, "unit": None, "avg_conf": None, "unit_conf": None, "tokens": []}

        orig_w, orig_h, arr = preprocess_rfdetr(dewarped_crop, CFG["digit_size"])
        boxes, scores, class_ids = decode_rfdetr(
            self.session, arr, orig_w, orig_h, CFG["digit_thr"]
        )
        result = self._assemble(boxes, scores, class_ids)
        print(f"  → Reading: {result['reading']}  (avg_conf={result['avg_conf']:.3f})")
        return result


# ──────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATOR
# ──────────────────────────────────────────────────────────────
class MeterPipeline:
    def __init__(self, save_crops=False):
        print("\n" + "═"*60)
        print("  Initialising Meter Reading Pipeline")
        print("═"*60)
        self.stage1 = StageDetection()
        self.stage2a = StageBrandClassifier()
        self.stage2b = StageSerialOCR()
        self.stage3  = StageDewarp()
        self.stage4  = StageDigitReading()
        self._save_crops = save_crops
        print("═"*60 + "\n")

    def run(self, img_pil, img_name="image"):
        t0 = time.time()
        print(f"\n{'─'*60}")
        print(f"  Processing: {img_name}")
        print(f"{'─'*60}")

        timings = {}

        # Stage 1 — Detect meter body + serial number region
        print("\n[Stage 1] Detection")
        _t = time.time(); s1 = self.stage1.run(img_pil); timings["s1_detection"] = round(time.time() - _t, 3)
        print(f"  ⏱  {timings['s1_detection']}s")

        # ── DEBUG: save all Stage 1 + Stage 3 crops ──────────────
        if self._save_crops:
            stem = Path(img_name).stem
            crop_dir = Path(CFG["output_dir"]) / "debug_crops" / stem
            crop_dir.mkdir(parents=True, exist_ok=True)
            if s1["meter_crop"] is not None:
                s1["meter_crop"].save(crop_dir / "s1_meter_body.jpg")
                print(f"  [CROP] Saved → {crop_dir / 's1_meter_body.jpg'}")
            else:
                print("  [CROP] s1_meter_body : None (not detected)")
            if s1["sn_crop"] is not None:
                s1["sn_crop"].save(crop_dir / "s1_serial_number.jpg")
                print(f"  [CROP] Saved → {crop_dir / 's1_serial_number.jpg'}")
            else:
                print("  [CROP] s1_serial_number : None (not detected)")

        # Stage 2A — Brand classification on meter body crop
        print("\n[Stage 2A] Brand Classification")
        _t = time.time(); s2a = self.stage2a.run(s1["meter_crop"]); timings["s2a_brand"] = round(time.time() - _t, 3)
        print(f"  ⏱  {timings['s2a_brand']}s")

        # Stage 2B — Serial number OCR on serial crop
        print("\n[Stage 2B] Serial Number OCR")
        _t = time.time(); s2b = self.stage2b.run(s1["sn_crop"]); timings["s2b_ocr"] = round(time.time() - _t, 3)
        print(f"  ⏱  {timings['s2b_ocr']}s")

        # Stage 3 — Dewarp meter body crop
        print("\n[Stage 3] Perspective Dewarp")
        _t = time.time(); s3 = self.stage3.run(s1["meter_crop"]); timings["s3_dewarp"] = round(time.time() - _t, 3)
        print(f"  ⏱  {timings['s3_dewarp']}s")

        # ── DEBUG: save Stage 3 dewarped crop ─────────────────────
        if self._save_crops:
            if s3["dewarped_crop"] is not None:
                label = "dewarped" if s3["keypoints"] is not None else "raw_fallback"
                s3["dewarped_crop"].save(crop_dir / f"s3_{label}.jpg")
                print(f"  [CROP] Saved → {crop_dir / f's3_{label}.jpg'}")
            else:
                print("  [CROP] s3_dewarped : None")

        # Stage 4 — Read digits from dewarped crop (fallback: meter_crop, then full image)
        print("\n[Stage 4] Digit Reading")
        digit_input = s3["dewarped_crop"] or s1["meter_crop"] or img_pil
        _t = time.time(); s4 = self.stage4.run(digit_input); timings["s4_digits"] = round(time.time() - _t, 3)
        print(f"  ⏱  {timings['s4_digits']}s")

        elapsed = time.time() - t0
        timings["total"] = round(elapsed, 3)

        result = {
            "image":          img_name,
            "elapsed_sec":    round(elapsed, 3),
            "timings":        timings,
            # Detection
            "meter_score":    s1["meter_score"],
            "sn_score":       s1["sn_score"],
            # Brand
            "brand":          s2a["brand"],
            "brand_conf":     s2a["brand_conf"],
            # Serial
            "serial_number":  s2b["serial_number"],
            "serial_conf":    s2b["serial_conf"],
            # Dewarp
            "dewarp_score":   s3["dewarp_score"],
            # Reading
            "reading":        s4["reading"],
            "unit":           s4["unit"],
            "unit_conf":      s4["unit_conf"],
            "avg_digit_conf": s4["avg_conf"],
            # Internals (for optional visualization)
            "_s1": s1,
            "_s3": s3,
            "_s4": s4,
        }

        self._print_summary(result)
        return result

    @staticmethod
    def _print_summary(r):
        print(f"\n{'═'*60}")
        print(f"  RESULT — {r['image']}")
        print(f"{'─'*60}")
        print(f"  Brand         : {r['brand']}  (conf={r['brand_conf']:.3f})" if r["brand"] else "  Brand         : N/A")
        print(f"  Serial Number : {r['serial_number']}  (conf={r['serial_conf']:.3f})" if r["serial_number"] else "  Serial Number : N/A")
        print(f"  Meter Reading : {r['reading']}")
        print(f"  Avg Digit Conf: {r['avg_digit_conf']:.3f}" if r["avg_digit_conf"] is not None else "  Avg Digit Conf: N/A")
        print(f"{'─'*60}")
        t = r.get("timings", {})
        print(f"  ⏱  Stage 1 Detection  : {t.get('s1_detection', 'N/A')}s")
        print(f"  ⏱  Stage 2A Brand     : {t.get('s2a_brand',    'N/A')}s")
        print(f"  ⏱  Stage 2B OCR       : {t.get('s2b_ocr',      'N/A')}s")
        print(f"  ⏱  Stage 3 Dewarp     : {t.get('s3_dewarp',    'N/A')}s")
        print(f"  ⏱  Stage 4 Digits     : {t.get('s4_digits',    'N/A')}s")
        print(f"  ⏱  TOTAL              : {r['elapsed_sec']}s")
        print(f"{'═'*60}\n")


# ──────────────────────────────────────────────────────────────
# OPTIONAL VISUALIZATION  (import matplotlib only if needed)
# ──────────────────────────────────────────────────────────────
def save_visualization(img_pil, result, save_path):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Patch

    s1 = result["_s1"]
    s3 = result["_s3"]
    s4 = result["_s4"]

    n_cols  = 3 if s3["dewarped_crop"] is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 7))
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#0d0d0d")

    # ── Col 0: original + detections ──
    ax = axes[0]
    ax.imshow(img_pil)
    ax.set_title("Stage 1 — Detection", color="white", fontsize=11)
    colors_det = {"meter": "#1D9E75", "serial": "#378ADD"}
    for box, score, cid in zip(s1["all_boxes"], s1["all_scores"], s1["all_class_ids"]):
        name  = CFG["detect_classes"][cid] if cid < len(CFG["detect_classes"]) else f"cls{cid}"
        color = colors_det.get(name.split("_")[0], "#888780")
        x1,y1,x2,y2 = box
        ax.add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=color, facecolor="none"))
        ax.text(x1, y1-5, f"{name}: {score:.2f}", color="white", fontsize=9,
                fontweight="bold", bbox=dict(facecolor=color, edgecolor="none", pad=2, alpha=0.85))

    # ── Col 1: dewarped / meter crop ──
    crop_img = s3["dewarped_crop"] if s3["dewarped_crop"] is not None else s1["meter_crop"]
    if crop_img is not None:
        ax = axes[1]
        ax.imshow(crop_img)
        title = "Stage 3 — Dewarped" if s3["keypoints"] is not None else "Stage 3 — Raw Crop (dewarp skipped)"
        ax.set_title(title, color="white", fontsize=11)

    # ── Col 2: digit detections (if dewarped available) ──
    if n_cols == 3 and crop_img is not None:
        ax  = axes[2]
        ax.imshow(crop_img)
        ax.set_title("Stage 4 — Digit Reading", color="white", fontsize=11)
        DCOLORS = {"digit": "#3cb44b", "unit": "#4363d8", "punct": "#f58231"}
        orig_w, orig_h = crop_img.size
        for tok in s4["tokens"]:
            x1,y1,x2,y2 = tok["box"]
            color = DCOLORS[tok["type"]]
            ax.add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                linewidth=2, edgecolor=color, facecolor=color, alpha=0.12))
            ax.add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                linewidth=2, edgecolor=color, facecolor="none"))
            ax.text((x1+x2)/2, y1-6, f"{tok['char']}\n{tok['conf']:.2f}",
                    fontsize=8, fontweight="bold", color="white", ha="center",
                    bbox=dict(facecolor=color, edgecolor="none", pad=2, alpha=0.9))
        legend = [Patch(facecolor=c, label=l) for l,c in DCOLORS.items()]
        ax.legend(handles=legend, loc="upper right", fontsize=8,
                  framealpha=0.85, facecolor="#1a1a2e", labelcolor="white")

    banner = (
        f"  Brand: {result['brand'] or 'N/A'}   |   "
        f"Serial: {result['serial_number'] or 'N/A'}   |   "
        f"Reading: {result['reading'] or 'N/A'}   |   "
        f"Elapsed: {result['elapsed_sec']}s  "
    )
    fig.text(0.5, 0.01, banner, ha="center", fontsize=11, fontweight="bold",
             color="white",
             bbox=dict(facecolor="#1a1a2e", edgecolor="#3cb44b",
                       boxstyle="round,pad=0.6", alpha=0.93))

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"  Visualization saved → {save_path}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def collect_images(path_str):
    p = Path(path_str)
    if p.is_file():
        return [p]
    return sorted([
        f for f in p.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    ])


def main():
    parser = argparse.ArgumentParser(description="Full Meter Reading Pipeline")
    parser.add_argument("--image",    required=True, help="Image file or folder path")
    parser.add_argument("--output",   default=CFG["output_dir"], help="Output directory")
    parser.add_argument("--save_vis",   action="store_true", help="Save annotated visualizations")
    parser.add_argument("--save_crops", action="store_true", help="Save debug crops for each stage")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = MeterPipeline(save_crops=args.save_crops)
    images   = collect_images(args.image)
    print(f"Processing {len(images)} image(s) …\n")

    all_results = []
    for img_path in images:
        img_pil = Image.open(img_path).convert("RGB")
        result  = pipeline.run(img_pil, img_name=img_path.name)

        # Strip internal stage data before saving JSON
        json_result = {k: v for k, v in result.items() if not k.startswith("_")}
        all_results.append(json_result)

        if args.save_vis:
            vis_path = out_dir / f"{img_path.stem}_result.jpg"
            save_visualization(img_pil, result, str(vis_path))

    # Save summary JSON
    summary_path = out_dir / "results.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✔  All done. JSON saved → {summary_path}")

    # Print final table
    print(f"\n{'═'*80}")
    print(f"  {'Image':<28} {'Brand':<14} {'Serial':<14} {'Reading':<18} {'Conf':>6}")
    print(f"{'─'*80}")
    for r in all_results:
        print(
            f"  {r['image']:<28}"
            f" {str(r['brand'] or 'N/A'):<14}"
            f" {str(r['serial_number'] or 'N/A'):<14}"
            f" {str(r['reading'] or 'N/A'):<18}"
            f" {r['avg_digit_conf']:>6.3f}" if r["avg_digit_conf"] is not None
            else f"  {r['image']:<28} {'N/A':<14} {'N/A':<14} {'N/A':<18} {'N/A':>6}"
        )
    print(f"{'═'*80}\n")


if __name__ == "__main__":
    main()