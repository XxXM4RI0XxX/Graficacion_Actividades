import math

import cv2
import time
import mediapipe as mp

def movement(pts, frame_x):

    # Mitad de la pantalla en eje X
    screen_mid = frame_x / 2

    # Distancia promedio entre todos los dedos
    dist_prom = (dist(pts[4], pts[8])+dist(pts[8], pts[12])+dist(pts[12], pts[16])+dist(pts[16], pts[20]))/4
    # Distancia entre la muñeca y el dedo medio
    dist_wrist_middle = dist(pts[0], pts[12])
    # Distancia entre la muñeca y el dedo meñique
    dist_wrist_pinky = dist(pts[0], pts[20])
    # Distancia entre el dedo medio y el índice
    dist_index_middle = dist(pts[8], pts[12])
    # Distancia entre el dedo medio y la primera articulación del dedo anular
    dist_middle_ring_dip = dist(pts[12], pts[15])
    # Verifica que la señal de movimiento sea correcta: Dedo indice y medio por encima de la mitad de la mano, lejos de los demás dedos
    move_verify = pts[16][0] > pts[9][0] and pts[20][0] > pts[9][0] and dist_middle_ring_dip > 90
    # Verificar solo meñique levantado
    only_pinky = pts[19][1] < pts[14][1] and pts[19][1] < pts[10][1] and pts[19][1] < pts[6][1]
    # Verificar solo indice levantado
    only_index = pts[6][1] < pts[10][1] and pts[6][1] < pts[14][1] and pts[6][1] < pts[19][1]

    # Zoom hacia dentro si todos los dedos están juntos
    if dist_prom < 25 and not only_pinky:
        print("Zoom out")
    # Zoom hacia afuera si todos los dedos están separados y lejos de la muñeca
    if dist_prom > 100 and dist_wrist_middle > 70.0 and dist_wrist_pinky > 70.0:
        print(dist_prom)
        print("Zoom in")
    # Habilita movimiento de la escena si el dedo índice y medio están juntos, y el verificador de movimiento es verdadero
    if dist_index_middle < 30.0 and move_verify:
        print("Moving")
        # Diferencia entre la mitad de la pantalla y la posición del dedo medio para velocidad de rotación
        move_dist = screen_mid - pts[12][0]
        print("Distancia: ",move_dist)
    if only_pinky and dist(pts[20],pts[15]) > 40 and dist_prom < 60:
        print(dist_prom)
        print("DOOWN")
    if only_index and dist(pts[7],pts[11]) > 30 and 55 < dist_prom < 80:
        print(dist_prom)
        print("UUUUUUP")


def dist(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )

MODEL_PATH = "C:\\Users\\monte\\Downloads\\hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# Conexiones de la mano (índices oficiales de MediaPipe)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # pulgar
    (0,5),(5,6),(6,7),(7,8),        # índice
    (5,9),(9,10),(10,11),(11,12),   # medio
    (9,13),(13,14),(14,15),(15,16), # anular
    (13,17),(17,18),(18,19),(19,20),(0,17) # meñique
]

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    start_time = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int((time.perf_counter() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            points = []
            for hand in result.hand_landmarks:
                for lm in hand:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    points.append((x, y))
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, points[a], points[b], (255, 0, 0), 2)

            movement(points, w)

        cv2.imshow("Hand Landmarker", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()
