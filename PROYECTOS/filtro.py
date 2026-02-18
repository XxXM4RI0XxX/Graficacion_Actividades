import cv2
import time
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "face_landmarker.task"

def make_mask(emotion="feli", size=512):

    img = np.zeros((size, size, 4), dtype=np.uint8)
    alpha = np.zeros((size, size), dtype=np.uint8)

    cx, cy = size // 2, size // 2
    axes = (int(size * 0.34), int(size * 0.49))

    # Rostro
    cv2.ellipse(alpha, (cx, cy), axes, 0, 0, 360, 255, -1)

    # Ojos
    eye_axes = (int(size * 0.10), int(size * 0.06))
    cv2.ellipse(alpha, (int(size * 0.38), int(size * 0.44)), eye_axes, 0, 0, 360, 0, -1)
    cv2.ellipse(alpha, (int(size * 0.62), int(size * 0.44)), eye_axes, 0, 0, 360, 0, -1)

    # Color
    if emotion == "feli":
        bgr = (0, 0, 255)
    else:
        bgr = (255, 0, 0)

    img[:, :, 0] = bgr[0]
    img[:, :, 1] = bgr[1]
    img[:, :, 2] = bgr[2]
    img[:, :, 3] = alpha

    # Borde
    cv2.ellipse(img, (cx, cy), axes, 0, 0, 360, (0, 0, 0, 120), 6, cv2.LINE_AA)

    # Nariz
    cv2.ellipse(img, (cx, int(size * 0.56)), (int(size * 0.04), int(size * 0.03)),
                0, 0, 360, (0, 0, 0, 160), 2, cv2.LINE_AA)

    # Boca
    mouth_center = (cx, int(size * 0.68))
    mouth_axes = (int(size * 0.16), int(size * 0.10))
    thickness = 10

    if emotion == "feli":
        # Sonrisa
        cv2.ellipse(img, mouth_center, mouth_axes, 0, 20, 160, (0, 0, 0, 255), thickness, cv2.LINE_AA)
    else:
        # Sonrisa'nt
        cv2.ellipse(img, (cx, int(size * 0.72)), mouth_axes, 0, 200, 340, (0, 0, 0, 255), thickness, cv2.LINE_AA)
        # Lagrima
        cv2.ellipse(img, (int(size * 0.40), int(size * 0.56)), (int(size * 0.015), int(size * 0.03)),
                    0, 0, 360, (255, 255, 255, 220), -1, cv2.LINE_AA)

    return img

# Sobre poner la mascará en la cámara
def overlay_rgba(vid, mask, ovlay_x, ovlay_y):
    sh, sw = mask.shape[:2]
    fh, fw = vid.shape[:2]

    # Verificar que la mácara no salga del video (recortar en caso de no salir totalmente)
    fx1, fy1 = max(0, ovlay_x), max(0, ovlay_y)
    fx2, fy2 = min(fw, ovlay_x + sw), min(fh, ovlay_y + sh)
    if fx1 >= fx2 or fy1 >= fy2:
        return

    # Recortar la parte de la máscara que si se muestra en video
    sx1, sy1 = fx1 - ovlay_x, fy1 - ovlay_y
    sx2, sy2 = sx1 + (fx2 - fx1), sy1 + (fy2 - fy1)

    sticker_crop = mask[sy1:sy2, sx1:sx2]

    alpha = sticker_crop[:, :, 3:4].astype(np.float32) / 255.0
    sticker_rgb = sticker_crop[:, :, :3].astype(np.float32)

    frame_crop = vid[fy1:fy2, fx1:fx2].astype(np.float32)

    vid[fy1:fy2, fx1:fx2] = ((1 - alpha) * frame_crop + alpha * sticker_rgb).astype(np.uint8)

# Convertir coordenadas de landmark a coordenada de pixel en pantalla
def px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


class Emotion:
    def __init__(self):
        self.mood = 0.0
        self.label = "traste"

    # Funciona mediante heurística (compara la altura entre las comisuras izquierda/derecha, y el centro de la boca para detectar la emoción)
    def update_from_landmarks(self, face, w, h):

        # Boca
        com_izq = px(face[61], w, h)     # comisura izq
        com_der = px(face[291], w, h)  # comisura der
        lab_sup = px(face[13], w, h)     # labio superior
        lab_inf = px(face[14], w, h)     # labio inferior

        # Promedia las coordenadas 'y' de las comisuras, obtiene el centro de la boca...
        corners_y = (com_izq[1] + com_der[1]) * 0.5
        mouth_center_y = (lab_sup[1] + lab_inf[1]) * 0.5

        # ... compara la si la altura de las comisuras en mayor o menor al centro de los labios
        raw = (mouth_center_y - corners_y) #/ eye_dist  # positivo => feli

        # Suavizado
        self.mood = 0.85 * self.mood + 0.15 * raw

        # Hysteresis para evitar parpadeo (Declara umbrales entre triste y feliz, para evitar cambios intermitentes)
        if self.mood > 2:
            self.label = "feli"
        elif self.mood < -1:
            self.label = "traste"

        return self.label, self.mood

# MAIN
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)

mask_feli = make_mask("feli", 512)
mask_traste = make_mask("traste", 512)

cap = cv2.VideoCapture(0)
state = Emotion()

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Espejo para evitar mareos
        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.perf_counter() - start) * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            face = result.face_landmarks[0]

            # Detectar emoción
            label, mood = state.update_from_landmarks(face, w, h)
            chosen_mask = mask_feli if label == "feli" else mask_traste

            # Dibujar máscara

            # Ojo izquierdo
            l_eye = px(face[33], w, h)
            # Ojo derecho
            r_eye = px(face[263], w, h)

            # Centro de la cara
            cx = (l_eye[0] + r_eye[0]) // 2
            cy = (l_eye[1] + r_eye[1]) // 2

            # Girar máscara junto al rostro (Ángulo de inclinación de una línea entre los ojos)
            angle = math.degrees(math.atan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))
            # Escala la mascára dependiendo de la distancia del rostro
            eye_dist = max(1.0, math.hypot(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))

            # Tamaño de máscara basado en distancia ojos (Relacionado con garantizar el tamaño correcto de la máscara en relación con la distancia del rostro)
            target_w = int(eye_dist * 2.35)
            target_h = int(target_w * (chosen_mask.shape[0] / chosen_mask.shape[1]))
            target_w = max(80, target_w)
            target_h = max(80, target_h)

            mask_resized = cv2.resize(chosen_mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            M = cv2.getRotationMatrix2D((target_w / 2, target_h / 2), -angle, 1.0)
            mask_rot = cv2.warpAffine(
                mask_resized, M, (target_w, target_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )

            # Coloca máscara un poco abajo del centro de ojos
            x = int(cx - target_w / 2)
            y = int(cy - target_h * 0.45)

            # Sobreponer la máscara sobre el video
            overlay_rgba(frame, mask_rot, x, y)

            # Valores de umbral
            cv2.putText(frame, f"{label} ={mood:+.3f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Traste X Feli", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')):
            break

cap.release()
cv2.destroyAllWindows()
