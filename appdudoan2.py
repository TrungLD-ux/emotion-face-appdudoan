import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from skimage.feature import hog
import os
from pathlib import Path # THÊM THƯ VIỆN PATHLIB QUAN TRỌNG NHẤT

# 1. ĐỊNH NGHĨA ĐƯỜNG DẪN GỐC AN TOÀN TRÊN SERVER
# Thư mục gốc là thư mục chứa file appdudoan.py này
BASE_DIR = Path(__file__).parent 

# =======================
# LOAD MODEL + SCALER
# =======================

# 1.1 Tải file chứa cả Model và thông tin. Tên file này phải có trên GitHub.
MODEL_FILE_NAME = "emotion_model.pkl"
try:
    emotion_model = joblib.load(BASE_DIR / MODEL_FILE_NAME)
    model = emotion_model["model"]
except FileNotFoundError:
    st.error(f"❌ Lỗi: Không tìm thấy file {MODEL_FILE_NAME}. Hãy kiểm tra trên GitHub!")
    st.stop()


# 1.2 KHẮC PHỤC LỖI SCALER!
# Thay vì lấy đường dẫn từ emotion_model (có thể là đường dẫn cục bộ), 
# chúng ta sẽ kiểm tra xem scaler có phải là một đối tượng (object) không, 
# hoặc giả định nó là 'scaler.pkl' nằm cùng thư mục (giả định phổ biến).

scaler = None
scaler_is_file = False

# Trường hợp 1: Scaler là một object được nhúng sẵn trong emotion_model (tốt nhất)
if "scaler" in emotion_model and emotion_model["scaler"] is not None:
    scaler = emotion_model["scaler"]
    
# Trường hợp 2: Scaler là một file riêng (chúng ta phải dùng đường dẫn an toàn)
else:
    SCALER_FILE_NAME = "scaler.pkl" # Đảm bảo file này có trên GitHub
    scaler_path = BASE_DIR / SCALER_FILE_NAME
    
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
            scaler_is_file = True
        except Exception as e:
            st.error(f"❌ Lỗi khi tải file {SCALER_FILE_NAME}: {e}")
            st.stop()

# Kiểm tra cuối cùng
if scaler is None:
    # Đoạn code này chỉ chạy nếu cả 2 trường hợp trên đều thất bại
    st.error(f"❌ Không tìm thấy scaler! Đảm bảo scaler.pkl có trên GitHub hoặc đã được nhúng vào emotion_model.pkl.")
    st.stop()

labels = emotion_model["labels"]

# =======================
# LOAD FACE DETECTOR (DNN)
# =======================
# 2. KHẮC PHỤC LỖI ĐƯỜNG DẪN TUYỆT ĐỐI!
# Thay thế đường dẫn tuyệt đối (E:\...) bằng đường dẫn tương đối an toàn.
# Giả định các file này nằm cùng thư mục gốc BASE_DIR.

CONFIG_FILE_NAME = "deploy.prototxt"
MODEL_FILE_NAME_DNN = "res10_300x300_ssd_iter_140000.caffemodel"

CONFIG_FILE = BASE_DIR / CONFIG_FILE_NAME
MODEL_FILE = BASE_DIR / MODEL_FILE_NAME_DNN

if not CONFIG_FILE.exists():
    st.error(f"Không tìm thấy file cấu hình DNN: {CONFIG_FILE_NAME}")
    st.stop()
if not MODEL_FILE.exists():
    st.error(f"Không tìm thấy file model DNN: {MODEL_FILE_NAME_DNN}")
    st.stop()

net = cv2.dnn.readNetFromCaffe(str(CONFIG_FILE), str(MODEL_FILE)) # Dùng str() để chuyển Path object sang string

# =======================
# STREAMLIT UI (GIỮ NGUYÊN)
# =======================
st.title("😄 Ứng dụng nhận diện cảm xúc khuôn mặt (HOG + DNN)")
st.write("Tải ảnh lên, hệ thống sẽ phát hiện khuôn mặt và dự đoán cảm xúc từng người.")

uploaded_file = st.file_uploader("📸 Chọn ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # ... (phần xử lý ảnh và dự đoán giữ nguyên như code cũ của bạn) ...

    # Giữ lại logic xử lý ảnh cũ của bạn
    img = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # === (1) Tiền xử lý ảnh để phát hiện mặt tốt hơn ===
    img_cv = cv2.convertScaleAbs(img_cv, alpha=1.2, beta=15)
    img_cv = cv2.GaussianBlur(img_cv, (3, 3), 0)

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

        # KIỂM TRA SCALER VÀ DỰ ĐOÁN
        if scaler is not None:
             if features.shape[1] != scaler.n_features_in_:
                st.warning(f"Số đặc trưng HOG không khớp: {features.shape[1]} vs {scaler.n_features_in_}")
                continue
             features_scaled = scaler.transform(features)
        else:
             features_scaled = features # Dùng features gốc nếu không tìm thấy scaler (chỉ nên dùng để debug)
             st.warning("Tiếp tục dự đoán mà không dùng Scaler.")

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