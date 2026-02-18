import cv2
import numpy as np
import math

video = cv2.VideoCapture(0)

canvas = None

# Colores (BGR)
paint_colors = {
    ord('1'): (0, 0, 255),     # rojo
    ord('2'): (0, 255, 0),     # verde
    ord('3'): (255, 0, 0),     # azul
    ord('4'): (0, 255, 255),   # amarillo
    ord('5'): (255, 0, 255)    # morado
}
current_color = (0, 0, 255)

brush_thickness = 8
min_area = 500

# Modo: 0 = libre, 1 = figuras
mode = 0

# Figura actual (en modo figuras)
shape_mode = 'square'  # q: square, w: circle, e: tri

# Para dibujar línea en modo libre
prev_pt = None

# ==========================
# DELAY / COOLDOWN DE FIGURAS
shape_delay_ms = 500  # <-- ajusta: 200-800 ms suele ir bien
last_shape_tick = 0   # tick del último "estampado"
freq = cv2.getTickFrequency()

def now_ms():
    # tiempo en ms usando ticks de OpenCV
    return (cv2.getTickCount() / freq) * 1000.0

def centroid_from_contour(c):
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def draw_square(img, p1, p2, color, thickness):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    side = max(abs(dx), abs(dy))
    if side < 10:
        return
    sx = 1 if dx >= 0 else -1
    sy = 1 if dy >= 0 else -1
    xB = x1 + sx * side
    yB = y1 + sy * side
    cv2.rectangle(img, (x1, y1), (xB, yB), color, thickness)

def draw_circle(img, p1, p2, color, thickness):
    x1, y1 = p1
    x2, y2 = p2
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    r = int(math.hypot(x2 - x1, y2 - y1) / 2)
    if r < 5:
        return
    cv2.circle(img, (cx, cy), r, color, thickness)

def draw_triangle(img, p1, p2, color, thickness):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L < 15:
        return

    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    px, py = -dy / L, dx / L
    h = (math.sqrt(3) / 2.0) * L

    x3 = int(mx + px * h)
    y3 = int(my + py * h)

    pts = np.array([[x1, y1], [x2, y2], [x3, y3]], dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

while video.isOpened():
    ret, vid = video.read()
    if not ret:
        break

    vid = cv2.flip(vid, 1)

    if canvas is None:
        h, w = vid.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    frame_hsv = cv2.cvtColor(vid, cv2.COLOR_BGR2HSV)

    # Azul
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(frame_hsv, lower_blue, upper_blue)

    # Limpieza de máscara
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    points = []
    for c in contours[:2]:
        pt = centroid_from_contour(c)
        if pt is not None:
            points.append(pt)

    for c in contours[:2]:
        cv2.drawContours(vid, [c], -1, (0, 255, 0), 2)

    # =========================
    # MODO 0: dibujo libre (1 mano)
    if mode == 0:
        if len(points) == 0:
            prev_pt = None
        else:
            cx, cy = points[0]
            cv2.circle(vid, (cx, cy), 6, (255, 255, 255), -1)

            if prev_pt is None:
                cv2.circle(canvas, (cx, cy), brush_thickness // 2, current_color, -1)
            else:
                cv2.line(canvas, prev_pt, (cx, cy), current_color, brush_thickness)
            prev_pt = (cx, cy)

    # =========================
    # MODO 1: figuras (2 manos) + DELAY
    else:
        prev_pt = None

        if len(points) >= 2:
            p1, p2 = points[0], points[1]
            cv2.circle(vid, p1, 7, (255, 255, 255), -1)
            cv2.circle(vid, p2, 7, (200, 200, 200), -1)
            cv2.line(vid, p1, p2, (255, 255, 255), 1)

            # Solo dibuja si ya pasó el cooldown
            t = now_ms()
            if (t - last_shape_tick) >= shape_delay_ms:
                if shape_mode == 'square':
                    draw_square(canvas, p1, p2, current_color, 3)
                elif shape_mode == 'circle':
                    draw_circle(canvas, p1, p2, current_color, 3)
                elif shape_mode == 'tri':
                    draw_triangle(canvas, p1, p2, current_color, 3)

                last_shape_tick = t

    overlay = cv2.addWeighted(vid, 0.75, canvas, 1.0, 0)

    mode_txt = "LIBRE (1 mano)" if mode == 0 else "FIGURAS (2 manos)"
    shape_txt = {"square": "QUAD(q)", "circle": "CIRC(w)", "tri": "TRIANG(e)"}[shape_mode]

    cv2.putText(
        overlay,
        f"Modo: {mode_txt} | Z MODE | FIG: {shape_txt} | 1-5 color | C clear | ESC salir | Delay: {shape_delay_ms}ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        2
    )

    cv2.imshow("Mask (blue)", mask)
    cv2.imshow("Canvas", canvas)
    cv2.imshow("Paint", overlay)

    key = cv2.waitKey(25) & 0xFF

    if key == 27:  # ESC
        break

    if key == ord('z') or key == ord('Z'):
        mode = 1 - mode

    if key in paint_colors:
        current_color = paint_colors[key]

    if key == ord('q'):
        shape_mode = 'square'
    elif key == ord('w'):
        shape_mode = 'circle'
    elif key == ord('e'):
        shape_mode = 'tri'

    if key == ord('c'):
        canvas[:] = 0
        prev_pt = None

video.release()
cv2.destroyAllWindows()
