import math
import cv2 as cv
import numpy as np
import random as ran

def function (R, r, t):
    res = ((R - r)/r)*t
    return res


#Definir dimension de la ventana
img_width,img_height = 800,800

#Crear ventana
img = np.zeros((img_height, img_width, 3), np.uint8)

#Conseguir centro de la imagen
x_i , y_i = img_width // 2, img_height // 2

#Escalar figura a mas grande
escalado = 2
#Declarar el ángulo theta inicial
t = 0
#
R = 1
#
r = 30
#
d = 60

while True:

    Rd,Bl,Gr = ran.randint(0,255),ran.randint(0,255),ran.randint(0,255)

    x = int(x_i - escalado * ((R - r) * math.cos(t) + d * math.cos(function(R, r, t))))
    y = int(y_i - escalado * ((R - r) * math.sin(t) + d * math.sin(function(R, r, t))))

    cv.circle(img, (x, y), 5, (Rd, Bl, Gr), -1)
    t += 3
    cv.imshow("image", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break