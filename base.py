import cv2
import mediapipe as mp
import numpy as np

# --- CONFIGURACIÓN DE DYSCRIMINA ---
# Inicializamos las herramientas de dibujo (para ver el esqueleto)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Configuración de la captura de video (0 suele ser la webcam por defecto)
cap = cv2.VideoCapture(0)

# Iniciamos el modelo de detección de posturas
# min_detection_confidence: Qué tan seguro debe estar para decir "esto es un cuerpo"
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    print(">>> DYSCRIMINA: SISTEMA DE VISIÓN ACTIVADO")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No se detecta la cámara.")
            continue

        # 1. PREPARACIÓN DE LA IMAGEN
        # MediaPipe necesita color RGB, pero OpenCV usa BGR. Convertimos.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # Mejora el rendimiento levemente
      
        # 2. INFERENCIA (El momento en que la máquina "piensa")
        results = pose.process(image)
      
        # 3. VISUALIZACIÓN
        # Volvemos a convertir a BGR para mostrarlo en pantalla
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Si encontró un cuerpo, dibuja el esqueleto
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), # Puntos (Articulaciones)
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)  # Líneas (Huesos)
            )
            
            # --- AQUÍ EMPEZARÁ LA DISCRIMINACIÓN ---
            # En el futuro, aquí leeremos las coordenadas para juzgar.
            # Por ahora, solo observamos.
            
        # Mostrar la ventana
        cv2.imshow('Dyscrimina - Fase 1: Percepcion', image)

        # Presiona 'q' para salir
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
