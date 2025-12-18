import cv2 as cv
import numpy as np
import random

h, w = 100, 150
r = 20

# Pelota A móvil
x, y = w // 2, h // 2
vx, vy = 2, 2

# Pelota B móvil'nt
bx, by = w // 4, h // 4


def collide(x1, y1, x2, y2, radius):
    # Distancia entre centros
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + dy * dy <= (2 * radius) * (2 * radius)


def random_free_spot(xx, yy, rr, avoid_x, avoid_y):
    # Busca lugares de manera aleatoria en la ventana, y verifica si no colisiona con la pelota móvil
    # Si el número de intentos por recolocarse supera 200, significa que noy hay lugar y simplemente no hace nada
    for i in range(200):
        nx = random.randint(rr, xx - rr)
        ny = random.randint(rr, yy - rr)

        dx = nx - avoid_x
        dy = ny - avoid_y

        # Verifica que la distancia entre pelotas sea la minima para evitar colisión
        if dx * dx + dy * dy > (2 * rr) * (2 * rr):
            return nx, ny
    return None


while True:
    img = np.ones((h, w, 3), np.uint8) * 255

    # Rebote en X
    if x + r >= w or x - r <= 0:
        vx *= -1

    # Rebote en Y
    if y + r >= h or y - r <= 0:
        vy *= -1

    x += vx
    y += vy

    # En caso de A chocar con B
    if collide(x, y, bx, by, r):
        spot = random_free_spot(w, h, r, x, y)
        if spot is not None:
            bx, by = spot

    cv.circle(img, (x, y), r, (255, 0, 0), -1)  # A: azul
    cv.circle(img, (bx, by), r, (0, 0, 255), -1)  # B: roja

    cv.imshow("Rebota pelota", img)
    if cv.waitKey(5) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
