import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from skimage.feature import hog
import os

# =======================
# LOAD MODEL + SCALER
# =======================
emotion_model = joblib.load("emotion_model.pkl")
model = emotion_model["model"]

# Nếu scaler lưu ở đường dẫn riêng
scaler_path = emotion_model.get("scaler_path", None)
if scaler_path and os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    st.error("❌ Không tìm thấy scaler! Kiểm tra lại file .pkl")
    st.stop()

labels = emotion_model["labels"]

# =======================
# LOAD FACE DETECTOR (DNN)
# =======================
CONFIG_FILE = r"E:\DudoanCamxuc\deploy.prototxt"
MODEL_FILE = r"E:\DudoanCamxuc\res10_300x300_ssd_iter_140000.caffemodel"

if not os.path.exists(CONFIG_FILE):
    st.error(f"Không tìm thấy file cấu hình: {CONFIG_FILE}")
    st.stop()
if not os.path.exists(MODEL_FILE):
    st.error(f"Không tìm thấy file model: {MODEL_FILE}")
    st.stop()

net = cv2.dnn.readNetFromCaffe(CONFIG_FILE, MODEL_FILE)

# =======================
# STREAMLIT UI
# =======================
st.title("😄 Ứng dụng nhận diện cảm xúc khuôn mặt (HOG + DNN)")
st.write("Tải ảnh lên, hệ thống sẽ phát hiện khuôn mặt và dự đoán cảm xúc từng người.")

uploaded_file = st.file_uploader("📸 Chọn ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # === (1) Tiền xử lý ảnh để phát hiện mặt tốt hơn ===
    img_cv = cv2.convertScaleAbs(img_cv, alpha=1.2, beta=15)  # tăng tương phản, sáng hơn
    img_cv = cv2.GaussianBlur(img_cv, (3, 3), 0)              # giảm nhiễu nhẹ

    (h, w) = img_cv.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img_cv, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # === (2) Giảm ngưỡng confidence để bắt được mặt nghiêng / bị che ===
        if confidence < 0.15:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # ====== CẮT KHUÔN MẶT ======
        face = img_cv[y1:y2, x1:x2]
        if face.size == 0:
            continue

        # === (3) Làm mịn vùng khuôn mặt để HOG ổn định ===
        face = cv2.resize(face, (96, 96))
        face = cv2.GaussianBlur(face, (3, 3), 0)

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)

        # ====== TRÍCH XUẤT HOG ======
        features, _ = hog(
            face_resized,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=True
        )
        features = features.reshape(1, -1)

        if features.shape[1] != scaler.n_features_in_:
            st.warning(f"Số đặc trưng HOG không khớp: {features.shape[1]} vs {scaler.n_features_in_}")
            continue

        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)[0]

        if isinstance(labels, list):
            try:
                label = labels[int(pred)]
            except:
                label = str(pred)
        else:
            label = str(pred)

        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_cv, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        count += 1

    if count == 0:
        st.warning("⚠️ Không tìm thấy khuôn mặt nào rõ ràng trong ảnh.")
    else:
        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB),
                 caption="Kết quả nhận diện cảm xúc",
                 use_container_width=True)
