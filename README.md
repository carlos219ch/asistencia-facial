# Control de Asistencia Facial (LBPH)

Proyecto de prácticas preprofesionales: sistema de control de asistencia mediante reconocimiento facial usando Python y OpenCV.

## Funcionalidades
- Detección de rostro en tiempo real (OpenCV Haar Cascade)
- Captura de dataset por usuario
- Entrenamiento de modelo LBPH (OpenCV Contrib)
- Reconocimiento facial en vivo con control de falsos positivos
- Registro de asistencia en CSV (una vez por día)

## Requisitos
- Python 3.x
- OpenCV Contrib (`opencv-contrib-python`)
- Visual Studio Code (opcional)

## Instalación
```bash
python -m venv .venv
## Windows:
.\.venv\Scripts\activate
pip install opencv-contrib-python

Uso
1.- Capturar dataset:
python src/capture_dataset.py

2.- Entrenar modelo:
python src/train_lbph.py

3.- Ejecutar reconocimiento:
python src/recognize_lbph.py

#NOTA
#El dataset (rostros) y archivos generados no se suben al repositorio por privacidad.