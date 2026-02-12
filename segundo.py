import cv2
import mediapipe as mp
import numpy as np

# --- CONFIGURACIÓN DE DYSCRIMINA ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

# Variables para "suavizar" el dato (que no tiemble tanto el número)
postura_promedio = 0

print(">>> DYSCRIMINA: Módulo de Juicio Activo.")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # 1. PERCEPCIÓN
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Obtener dimensiones de la pantalla (Alto y Ancho)
        h, w, _ = image.shape

        # 2. EXTRACCIÓN DE DATOS (Aquí empieza la vigilancia real)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Extraer coordenadas de puntos clave
            # Multiplicamos por w (ancho) y h (alto) porque MediaPipe da datos entre 0 y 1
            nariz = [landmarks[mp_pose.PoseLandmark.NOSE.value].x * w,
                     landmarks[mp_pose.PoseLandmark.NOSE.value].y * h]
            
            hombro_izq = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
            
            hombro_der = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]

            # 3. CÁLCULO DE LA METRICA (La decisión matemática)
            # Calculamos el punto medio entre los hombros
            centro_hombros_y = (hombro_izq[1] + hombro_der[1]) / 2
            
            # Distancia vertical: Qué tan lejos está la nariz de la línea de los hombros
            # Si el valor es bajo, estás encorvada. Si es alto, cuello estirado.
            distancia_postura = centro_hombros_y - nariz[1]

            # 4. MANIFESTACIÓN (Dashboard en vivo)
            # Dibujamos una barra de "Nivel de Dominancia"
            cv2.rectangle(image, (50, 200), (100, 400), (0, 0, 0), -1) # Fondo barra
            
            # Lógica simple de clasificación (Sesgo programado)
            color_status = (0, 255, 0) # Verde por defecto
            etiqueta = "NEUTRO"
            
            if distancia_postura < 80: # Umbral arbitrario de "mala postura"
                color_status = (0, 0, 255) # Rojo
                etiqueta = "SUMISO"
            elif distancia_postura > 130:
                color_status = (255, 255, 0) # Cyan
                etiqueta = "DOMINANTE"

            # Visualizar métricas en pantalla
            cv2.putText(image, f"INDICE: {int(distancia_postura)}", (50, 180), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            cv2.putText(image, f"ESTADO: {etiqueta}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_status, 3)

            # Dibujar esqueleto normal
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow('Dyscrimina - Fase 2: Juicio', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()