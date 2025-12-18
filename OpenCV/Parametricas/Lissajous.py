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
escalado = 1
#Declarar el ángulo theta inicial
t = 0
# A = Ancho de figura
# B = Alto de figura
A , B = 200,200
# a , b = Modificar ejes
a , b = 3, 2
omega = math.radians(math.pi/2)

while True:

    x = int(x_i - escalado * (A * math.sin(a*t + omega)))
    y = int(y_i - escalado * (B * math.sin(b*t)))

    cv.circle(img, (x, y), 5, (255, 255, 255), -1)
    t += 0.01
    cv.imshow("image", img)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break