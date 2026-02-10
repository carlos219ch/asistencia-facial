import csv
import os
from datetime import datetime

CSV_PATH = "output/asistencia.csv"

def ensure_csv_exists(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.isfile(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["nombre", "fecha", "hora", "confidence"])

def already_marked_today(name: str, date_str: str) -> bool:
    """Retorna True si ya existe un registro de 'name' en 'date_str' (YYYY-MM-DD)."""
    if not os.path.isfile(CSV_PATH):
        return False

    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("nombre") == name and row.get("fecha") == date_str:
                return True
    return False

def mark_attendance_once_per_day(name: str, confidence: float) -> bool:
    """
    Registra asistencia solo si NO existe un registro para hoy.
    Retorna True si registró, False si ya estaba registrado.
    """
    ensure_csv_exists(CSV_PATH)
    now = datetime.now()
    fecha = now.strftime("%Y-%m-%d")
    hora = now.strftime("%H:%M:%S")

    if already_marked_today(name, fecha):
        return False

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([name, fecha, hora, f"{confidence:.1f}"])
    return True
