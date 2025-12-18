import math
import cv2 as cv
import numpy as np

#Definir dimension de la ventana
img_width,img_height = 800,800

#Crear ventana
img = np.zeros((img_height, img_width, 3), np.uint8)

#Conseguir centro de la imagen
x_i , y_i = img_width // 2, img_height // 2

#Escalar figura a mas grande
escalado = 10
#Declarar el ángulo theta inicial
t = 0

while True:

    x = int(x_i - escalado * (16*(math.sin(t)**3)))
    y = int(y_i - escalado * (13*math.cos(t) - 5*math.cos(2*t) - 2*math.cos(3*t) - math.cos(4*t)))

    cv.circle(img, (x, y), 1, (255, 255, 255), -1)
    t -= 0.1
    cv.imshow("image", img)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break