import os
import cv2
import time
from attendance_csv import mark_attendance_once_per_day

# =========================
# CONFIGURACIÓN
# =========================
MODEL_PATH = "models/lbph_model.yml"
LABELS_PATH = "models/labels.txt"

THRESHOLD = 55          # umbral estricto (ajustado por ti)
HITS_REQUIRED = 3       # frames consecutivos para confirmar
COOLDOWN_SECONDS = 30   # no repetir asistencia en X segundos
IMG_SIZE = (200, 200)

# =========================
# UTILIDADES
# =========================
def load_labels(path: str):
    id_to_name = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                i, name = line.strip().split(",", 1)
                id_to_name[int(i)] = name
    return id_to_name

# =========================
# MAIN
# =========================
def main():
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(LABELS_PATH):
        raise RuntimeError("❌ Falta el modelo. Ejecuta primero: python src\\train_lbph.py")

    id_to_name = load_labels(LABELS_PATH)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ No se pudo abrir la cámara.")

    print("Reconocimiento OK. Q para salir.")

    # =========================
    # VARIABLES DE ESTADO
    # =========================
    hits = 0
    last_mark_time = 0.0
    last_mark_name = None
    attendance_marked = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=7, minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, IMG_SIZE)

            label_id, confidence = recognizer.predict(face)
            name = id_to_name.get(label_id, "Desconocido")

            # =========================
            # CONFIRMACIÓN POR FRAMES
            # =========================
            if confidence <= THRESHOLD and name != "Desconocido":
                hits += 1
            else:
                hits = max(0, hits - 1)

            if hits >= HITS_REQUIRED:
                name_to_show = name
            else:
                name_to_show = "Desconocido"

            # =========================
            # REGISTRO DE ASISTENCIA
            # =========================
            now = time.time()
            if name_to_show != "Desconocido" and not attendance_marked:
                did_mark = mark_attendance_once_per_day(name_to_show, confidence)

                if did_mark:
                    print(f"✅ Asistencia registrada HOY: {name_to_show} ({confidence:.1f})")
                else:
                    print(f"ℹ️ Ya estaba registrado HOY: {name_to_show}")
                attendance_marked = True  # evita spamear mensajes mientras sigues frente a la cámara

            # =========================
            # DIBUJO EN PANTALLA
            # =========================
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{name_to_show} ({confidence:.1f})",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        cv2.putText(
            frame,
            f"Umbral: {THRESHOLD} | Q: salir",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("Asistencia Facial - LBPH", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
