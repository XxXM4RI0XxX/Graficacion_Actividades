import cv2 as cv
import numpy as np

# Detectar click y hold del cursor dentro de la ventana
def mouse_event(event, mx, my, flags, img):
    global color, mouse_down, brush_size

    if event == cv.EVENT_LBUTTONDOWN:
        mouse_down = True

        # Selección de color (solo si click fue en la barra)
        if 10 <= mx <= xx - 5 and 10 <= my <= yy:
            color = (0, 0, 0)
        elif xx <= mx <= (xx * 2) - 5 and 10 <= my <= yy:
            color = (255, 0, 0)
        elif xx * 2 <= mx <= (xx * 3) - 5 and 10 <= my <= yy:
            color = (0, 255, 0)
        elif xx * 3 <= mx <= (xx * 4) - 5 and 10 <= my <= yy:
            color = (0, 0, 255)
        elif xx * 4 <= mx <= (xx * 5) - 5 and 10 <= my <= yy:
            color = (0, 255, 255)
        elif xx * 5 <= mx <= (xx * 6) - 5 and 10 <= my <= yy:
            color = (255, 255, 255)

        elif my > yy + 10:
            cv.circle(img, (mx, my), brush_size, color, -1)

    elif event == cv.EVENT_LBUTTONUP:
        mouse_down = False

    elif event == cv.EVENT_MOUSEMOVE and mouse_down:
        if my > yy + 10:
            cv.circle(img, (mx, my), brush_size, color, -1)




x , y = 500, 500

img = np.ones((x,y,3), np.uint8) * 255

xx = int(x/6)
yy = int(y/10)

# INTERFAZ

# Linea divisora
img = cv.line(img, (0,yy+10), (x,yy+10), 0, 2)

# Negro
img = cv.rectangle(img, (10, 10), (xx-5,yy), 0.0, -1)
# Azul
img = cv.rectangle(img, (xx, 10), ((xx*2)-5,yy), (255,0,0), -1)
# Verde
img = cv.rectangle(img, (xx*2, 10), ((xx*3)-5,yy), (0,255,0), -1)
# Rojo
img = cv.rectangle(img, (xx*3, 10), ((xx*4)-5,yy), (0,0,255), -1)
# Amarillo
img = cv.rectangle(img, (xx*4, 10), ((xx*5)-5,yy), (0,255,255), -1)
# Blanco
img = cv.rectangle(img, (xx*5, 10), ((xx*6)-5,yy), (0,0,0), 1)

color = (0,0,0)
brush_size = 5

mouse_down = False

cv.namedWindow("Ventana")
cv.setMouseCallback("Ventana", mouse_event,img)

while True:
    cv.imshow("Ventana", img)
    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()