# ⚡ SureScanOCR — Meter Reading Pipeline

A **high-precision, multi-stage deep learning pipeline** for automated meter reading from images. Combines object detection, classification, OCR, and perspective correction to extract meter readings with confidence scores.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python) 
![Models](https://img.shields.io/badge/Models-ONNX%20%26%20PyTorch-green)

<img width="950" height="407" alt="image" src="https://github.com/user-attachments/assets/c5f2d163-7a6a-4604-9946-213c3c7f57f6" />


<img width="898" height="363" alt="image" src="https://github.com/user-attachments/assets/5fdadcc8-8b01-46cd-8564-2aeb85e6c4b4" />


<img width="949" height="307" alt="image" src="https://github.com/user-attachments/assets/80106220-c579-40c6-9c6c-8cf765a8b173" />


<img width="953" height="350" alt="image" src="https://github.com/user-attachments/assets/ce87654c-4821-4fa5-b5f1-56447143be92" />



## 🎯 Features

- **📸 4-Stage Inference Pipeline** — Detection → Classification → OCR → Digit Reading
- **🎯 Meter Detection** — RF-DETR model for precise meter body and serial number localization
- **🏷️ Brand Classification** — EfficientNet-B0 for meter brand identification  
- **🔤 Serial Number OCR** — CTC-based OCR for automatic serial number extraction
- **🔧 Perspective Correction** — Keypoint RCNN for intelligent meter face dewarp
- **📊 Digit Recognition** — Advanced digit detection with confidence per token
- **⏱️ Performance Metrics** — Per-stage timing breakdown for optimization
- **🎨 Interactive Dashboard** — Streamlit web UI with real-time visualization
- **💾 Debug Crops & Visualization** — Save intermediate processing stages for analysis

---

## 🛠️ Architecture

### **Stage 1: Meter Detection** 
- Model: `RF-DETR ONNX` 
- Detects meter body and serial number region
- Outputs: meter crop, serial crop with confidence scores

### **Stage 2A: Brand Classification**
- Model: `EfficientNet-B0 PyTorch`
- Classifies meter brand from detected meter crop
- Runs in parallel with Stage 2B

### **Stage 2B: Serial Number OCR**
- Model: `CTC ONNX (PaddleOCR)`
- Extracts alphanumeric serial number via CTC decoding
- Runs in parallel with Stage 2A

### **Stage 3: Perspective Dewarp**
- Model: `Keypoint RCNN TorchScript`
- Detects 4 corner keypoints of meter display
- Applies perspective transform for frontal view

### **Stage 4: Digit Reading**
- Model: `RF-DETR ONNX`
- Detects individual digits, punctuation, and units
- Assembles reading with confidence per token

---

## 📋 Requirements

```
Python 3.8+
PyTorch / TorchVision
ONNX Runtime
PaddleOCR (implied, dependencies in pipeline)
EfficientNet (timm)
Streamlit
Pillow
NumPy
OpenCV (cv2)
```

See `req.txt` for the full dependency list (or create one from requirements below).

---

## 🚀 Quick Start

### 1. **Setup**

```bash
# Clone the repository
git clone https://github.com/Kush-tech1/SureScanOCR.git
cd SureScanOCR

# Install dependencies
pip install torch torchvision timm onnxruntime pillow opencv-python streamlit numpy
```

### 2. **Configure Model Paths**

Edit `pipeline.py` and update the `CFG` dictionary with your model file paths:

```python
CFG = {
    "detection_model":    "path/to/inference_model_working_sn+mbody.onnx",
    "classifier_model":   "path/to/meter_classifier.pth",
    "ocr_model":          "path/to/sn_ocr.onnx",
    "ocr_dict":           "path/to/dict.txt",
    "dewarp_model":       "path/to/deploy_kp_model.pt",
    "digit_model":        "path/to/inference_model_digits.onnx",
    # ...thresholds and other settings
}
```

### 3. **Run via CLI**

```bash
# Single image
python pipeline.py --image meter.jpg

# Folder of images
python pipeline.py --image ./meter_images/

# Save visualizations
python pipeline.py --image meter.jpg --save_vis

# Save intermediate crops for debugging
python pipeline.py --image meter.jpg --save_crops
```

**Output:** Results saved to `pipeline_results/results.json` (and visualizations if `--save_vis` is set)

### 4. **Run Interactive Dashboard**

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser:
- 📤 Upload meter images
- 🎚️ Adjust detection thresholds in real-time
- 📊 View meter reading, brand, serial, and timing breakdown
- 🔍 Inspect intermediate crops (Stage 1, 3, 4)
- 💾 Download results

---

## 📊 Output Format

### JSON Results

```json
{
  "image": "meter_001.jpg",
  "reading": "12345 kWh",
  "brand": "ABC Meter",
  "serial_number": "SN123456",
  "brand_conf": 0.95,
  "serial_conf": 0.92,
  "meter_score": 0.87,
  "sn_score": 0.85,
  "dewarp_score": 0.88,
  "avg_digit_conf": 0.93,
  "unit": "kWh",
  "elapsed_sec": 2.341,
  "timings": {
    "s1_detection": 0.512,
    "s2a_brand": 0.234,
    "s2b_ocr": 0.198,
    "s3_dewarp": 0.321,
    "s4_digits": 0.456,
    "total": 2.341
  }
}
```

### Visualization Output

Annotated images with:
- **Col 1:** Original image + detection boxes
- **Col 2:** Dewarped meter display
- **Col 3:** Digit detections with confidence overlay

---

## ⚙️ Configuration Guide

### Thresholds (in `CFG`)

| Parameter | Range | Default | Use Case |
|-----------|-------|---------|----------|
| `detect_thr` | 0.1 — 1.0 | 0.30 | Lower if meter body/SN not detected |
| `dewarp_thr` | 0.1 — 1.0 | 0.50 | Lower for angled/low-quality images |
| `digit_thr` | 0.1 — 1.0 | 0.70 | Adjust for digit recognition sensitivity |

### Input Sizes

- **Detection models:** 512×512 (auto-resized)
- **Classifier:** 224×224
- **OCR input:** 3×48×320 (normalized)

---

## 🎓 Class Mappings

### Stage 1 Classes
```python
["display_kp", "SN", "meter_body"]
```

### Stage 4 Digit Classes
```python
[".", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
 "KVah", "Kw", "MD", "colon", "kVA"]
```

---

## 🔍 Debugging

### Enable Debug Crops
```bash
python pipeline.py --image meter.jpg --save_crops
```
Crops saved to `pipeline_results/debug_crops/`

### Check Class Detection
The pipeline prints class-wise detection counts:
```
[DEBUG] Raw detections above thr: 3
[DEBUG]   class_id=0 name='display_kp'  count=1  top_score=0.921
[DEBUG]   class_id=1 name='SN'          count=1  top_score=0.887
[DEBUG]   class_id=2 name='meter_body'  count=1  top_score=0.945
```

### Streamlit Interactive Debugging
Use the sidebar in `app.py` to:
- Adjust thresholds live
- Toggle crop visualization
- Monitor per-stage timings

---

## 📈 Performance

Typical performance on NVIDIA GPU:
- **Full pipeline:** ~2.3s per image
- **Stage 1 (Detection):** 0.5s
- **Stage 2A (Brand):** 0.2s
- **Stage 2B (OCR):** 0.2s
- **Stage 3 (Dewarp):** 0.3s
- **Stage 4 (Digits):** 0.5s

*Times vary with image resolution and hardware.*

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit improvements
4. Submit a pull request

---

## 📝 License

This project is provided as-is. Ensure all model licenses are respected.

---

## 📧 Support

For issues, questions, or feedback:
- Open a GitHub issue
- Check existing issues for solutions
- Review stage-specific documentation above

---

## 🏆 Acknowledgments

Built with:
- **RF-DETR** — Object detection backbone
- **EfficientNet-B0** — Efficient classification
- **PaddleOCR / SVTR-HGNet** — OCR models
- **Keypoint RCNN** — Perspective detection
- **Streamlit** — Interactive UI

---

**Made with ❤️ for accurate meter reading automation**
