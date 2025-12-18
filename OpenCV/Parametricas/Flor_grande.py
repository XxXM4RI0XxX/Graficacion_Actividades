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
escalado = 20
#Declarar el ángulo theta
theta = 0
# a = numero de petalos
a = 10

while True:

    x = int(x_i + escalado * (a * math.cos(theta) + math.cos(a * theta)))
    y = int(y_i + escalado * (a * math.sin(theta) + math.sin(a * theta)))

    cv.circle(img, (x, y), 2, (255, 255, 255), -1)
    theta += 0.1
    cv.imshow("image", img)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break