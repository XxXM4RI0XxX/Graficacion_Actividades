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
escalado = 100
#Declarar el ángulo theta inicial
theta = 0
#Orientacion del espiral
inverso = True
# a = cercanía del inicio al centro de la figura
# b = ancho de onda
a , b = 10, 0.05

while True:

    x = int(x_i + escalado * (a * math.e**(b*theta) * math.cos(theta)))
    y = int(y_i + escalado * (a * math.e**(b*theta)) * math.sin(theta))

    cv.circle(img, (x, y), 1, (255, 255, 255), -1)
    if not inverso:
        theta += 0.1
    else:
        theta -= 0.1
    cv.imshow("image", img)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break