import os
import cv2
import numpy as np

DATASET_DIR = "dataset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lbph_model.yml")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")

IMG_SIZE = (200, 200)

def load_faces_and_labels(dataset_dir: str):
    faces = []
    labels = []
    label_map = {}  # name -> id
    current_id = 0

    if not os.path.isdir(dataset_dir):
        raise RuntimeError(f"No existe la carpeta '{dataset_dir}'. Crea dataset/<Usuario>/ con fotos.")

    for person_name in sorted(os.listdir(dataset_dir)):
        person_path = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        if person_name not in label_map:
            label_map[person_name] = current_id
            current_id += 1

        person_id = label_map[person_name]

        for file in os.listdir(person_path):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, IMG_SIZE)

            faces.append(gray)
            labels.append(person_id)

    if len(faces) == 0:
        raise RuntimeError("No se encontraron imágenes. Asegúrate de tener fotos en dataset/<Usuario>/")

    return np.array(faces), np.array(labels), label_map

def save_label_map(label_map: dict, path: str):
    # guarda: id,name
    inv = sorted([(v, k) for k, v in label_map.items()], key=lambda x: x[0])
    with open(path, "w", encoding="utf-8") as f:
        for i, name in inv:
            f.write(f"{i},{name}\n")

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    faces, labels, label_map = load_faces_and_labels(DATASET_DIR)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)

    recognizer.save(MODEL_PATH)
    save_label_map(label_map, LABELS_PATH)

    print("✅ Entrenamiento completado")
    print(f"Modelo: {MODEL_PATH}")
    print(f"Labels: {LABELS_PATH}")
    print(f"Personas: {list(label_map.keys())}")
    print(f"Imágenes usadas: {len(faces)}")

if __name__ == "__main__":
    main()
