import cv2
import os
from datetime import datetime

PERSON_NAME = "Carlos"   # <-- cambia el nombre si quieres
MAX_PHOTOS = 20          # cantidad de fotos a capturar

def main():
    # Carpeta destino
    out_dir = os.path.join("dataset", PERSON_NAME)
    os.makedirs(out_dir, exist_ok=True)

    # Detector de rostro
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    count = 0
    print("Presiona S para guardar una foto del rostro (recortada). Q para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        # Elegir el rostro más grande (si hay varios)
        face_box = None
        if len(faces) > 0:
            face_box = max(faces, key=lambda b: b[2] * b[3])  # (x,y,w,h)

            x, y, w, h = face_box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(frame, f"Guardadas: {count}/{MAX_PHOTOS}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "S: guardar  Q: salir", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Captura de dataset", frame)

        key = cv2.waitKey(1) & 0xFF

        # Guardar
        if key == ord("s"):
            if face_box is None:
                print("No se detecto rostro. Intenta de nuevo.")
                continue

            x, y, w, h = face_box
            face_crop = frame[y:y+h, x:x+w]

            # Normalizar tamaño (útil para entrenamiento)
            face_crop = cv2.resize(face_crop, (200, 200))

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(out_dir, f"{PERSON_NAME}_{ts}.jpg")

            cv2.imwrite(filename, face_crop)
            count += 1
            print(f"[{count}] Guardado: {filename}")

            if count >= MAX_PHOTOS:
                print("Listo: dataset completado.")
                break

        # Salir
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
