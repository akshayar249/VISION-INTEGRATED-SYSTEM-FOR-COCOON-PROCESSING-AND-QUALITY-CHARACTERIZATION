# VISION-INTEGRATED-SYSTEM-FOR-COCOON-PROCESSING-AND-QUALITY-CHARACTERIZATION
# Vision-Integrated System for Cocoon Processing and Quality Characterization

An end-to-end **automated cocoon detection, quality classification, and sorting system** designed for small and medium-scale sericulture units. This project integrates **embedded systems, sensors, computer vision (YOLOv8), and mechanical actuation** to reduce manual labor and improve silk quality before reeling.

---

## 📌 Problem Statement

Traditional cocoon sorting in sericulture is manual, time-consuming, and inconsistent. Defective cocoons entering the reeling process cause frequent filament breakage and reduced silk yield. Existing automated solutions are often expensive and inaccessible to small-scale units.

This project proposes a **low-cost, vision-integrated automation system** for real-time cocoon grading and sorting.

---

## 🎯 Objectives

* Real-time cocoon image capture using a controlled tunnel-based imaging setup
* Automatic classification of cocoons as **Good** or **Defective** using a YOLOv8 model
* Dataset enhancement using **Convolutional Autoencoder (CAE)** based augmentation
* Automated mechanical sorting using servo motors
* Reduction of human error, labor dependency, and processing time

---

## 🧠 System Overview

**Workflow:**

1. **Loader Module** – IR sensor detects cocoon entry and controls a gate mechanism
2. **Tunnel & Deflosser** – Aligns cocoon and removes loose floss using rotating brushes
3. **Imaging Setup** – Camera captures uniform, high-quality cocoon images
4. **YOLOv8 Model** – Detects and classifies cocoons (Good / Bad)
5. **Communication Layer** – Flask server bridges ML model and microcontroller
6. **Sorting Mechanism** – Servo motor routes cocoon into appropriate bin

---

## 🧩 Hardware Components

| Component                    | Purpose                    |
| ---------------------------- | -------------------------- |
| Arduino UNO / ESP32          | Gate & servo control       |
| IR Beam Sensor               | Cocoon detection           |
| Servo Motors (SG90 / MG995)  | Sorting and gate actuation |
| Intel RealSense / USB Camera | Image capture              |
| LED Panel (15W)              | Uniform illumination       |
| Deflosser Mechanism          | Remove loose silk floss    |
| Tunnel Assembly              | Stable cocoon movement     |
| 5V Power Supply              | System power               |

---

## 💻 Software Stack

* **Python 3.11**
* **YOLOv8 (Ultralytics)**
* **PyTorch**
* **OpenCV**
* **Flask (Local Server)**
* **Google Colab (Model Training & Inference)**
* **Arduino IDE**

---

## 📊 Dataset & Training

* **Initial Dataset:** 305 manually labeled cocoon images
* **Augmentation:** Rotation, flipping, brightness changes + CAE-generated synthetic images
* **Expanded Dataset:** ~689 images
* **Train / Val / Test Split:** 70% / 20% / 10%

**Model Details:**

* YOLOv8 (Detection + Binary Classification)
* Optimizer: AdamW
* Epochs: 300
* Real-time inference: < 100 ms per frame

---

## ⚙️ Sorting Logic

* **Good Cocoon → Servo at 45°**
* **Defective Cocoon → Servo at 135°**

Sorting response time: < 1 second

---

## ✅ Results

* High precision and recall in Good vs Bad cocoon classification
* Reliable real-time performance under variable lighting
* Consistent feeding and sorting without overlaps
* Significant reduction in manual inspection effort

---

## 🚀 Future Enhancements

* Multi-class cocoon grading (double, pierced, immature, flimsy)
* Larger and more diverse datasets
* Edge-device deployment (Jetson / Raspberry Pi)
* Conveyor synchronization for higher throughput
* Wireless monitoring dashboard and analytics

---

## 👩‍💻 Team

**Department of Electronics and Communication Engineering**
BMS Institute of Technology and Management, Bangalore

* **Akshaya Ramesh** – 1BY23EC009
* **Ashitha M** – 1BY23EC024
* **Raman R** – 1BY23EC085
* **Vibha I S** – 1BY23EC122

---

## 📜 License

This project is developed for academic and research purposes. Feel free to fork and adapt with proper attribution.

---

⭐ If you find this project useful, consider giving the repository a star!
