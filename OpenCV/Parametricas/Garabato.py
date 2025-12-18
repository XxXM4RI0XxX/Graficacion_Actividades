import math

import cv2 as cv
import numpy as np

#Definir dimension de la ventana
img_width,img_height = 800,800

#Crear ventana
img = np.zeros((img_height, img_width, 3), np.uint8)

#Conseguir centro de la imagen
x_i , y_i = img_width // 2, img_height // 2

#Declarar el ángulo theta
theta = 0

while True:

    # Ecuación de la paramétrica
    # Se hace el casteo a dato entero porque cv.circle solo acepta enteros
    # para la coordenada central
    # Al ser una paramétrica, el unico valor que cambia es el ángulo theta
    # por lo que, el punto central nunca cambia, y solo se le suma el resultado
    # de la ecuación
    # Cada iteración del ciclo, el valor del ángulo se cambia, y nuevo punto
    # en la ecuación paramétrica es graficado
    # La multiplicación 100 * Ec..., sirve para escalar la animación
    x = int(x_i + 100*(math.cos(3 * theta) - math.cos(2 * theta)**5))
    y = int(y_i + 100*(math.sin(4 * theta) - math.sin(3 * theta)**2))

    cv.circle(img, (x, y), 2, (255, 255, 255), -1)
    theta += 0.01
    cv.imshow("image", img)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break