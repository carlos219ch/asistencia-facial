import cv2

def main():
    cap = cv2.VideoCapture(0)

    # Si tu cámara no abre, prueba 1 o 2
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara (índice 0/1).")

    print("Cámara OK. Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer un frame de la cámara.")
            break

        cv2.imshow("Asistencia Facial - Camera Test", frame)

        # salir con q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
