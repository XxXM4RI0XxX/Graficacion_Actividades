import cv2 as cv
import numpy as np

#Crear imagen * escala de gris
color_fondo = 200
img = np.ones((500,500,3), np.uint8) * color_fondo

#Poligono del carro
cuerpo = np.array([[30, 240], [40, 300],  #Defensa trasera
                   [100,320], [460,320],  #Parte baja
                   [480,245],  #Defensa delantera
                   [440,230], [330,215],  #Cofre
                   [285,180], [250,160], [130,170]]) #Techo
cuerpo = cuerpo.reshape((-1, 1, 2))
#Rellenar el poligono
cv.fillPoly(img, [cuerpo], (220, 50, 10))
#Faro
cv.rectangle(img, (440,230), (490,260), (0,220,220),-1)
cv.line(img, (440,230),(440,260),(0,0,0),2)
cv.line(img, (490,260),(440,260),(0,0,0),2)
#Contorno del poligono: polylines(img, [arreglo_pts], cerrar_primer_ultimo_pto, (r,g,b), grosor)
cv.polylines(img, [cuerpo], True, (0, 0, 0), 2)

#Ventanas
ventana = np.array([[320,215],[280, 185], [250, 167], [145, 177], [95, 215]])
cv.polylines(img, [ventana], True, (0, 0, 0), 3)
cv.fillPoly(img,[ventana], (255,255,0))

#Puertas
puerta = np.array([[145,217],[145,260],[190,310],[322,310],[322,215]])
cv.polylines(img, [puerta], False, (0,0,0),2)

#Divisiones ventanas-puertas
cv.line(img, (puerta[0]),(ventana[3]),(0,0,0),4)
#manija
cv.line(img, (165,240),(190,240),(0,0,0),4)

#Rueda 1
cc1 = 120
cv.circle(img, (cc1, 310), 45, (0, 0, 0), -1) #LLanta
cv.circle(img, (cc1, 310), 25, (100, 100, 100), -1) #Rin exterior
cv.circle(img, (cc1, 310), 10, (150, 150, 150), -1) #Rin interior

#Rueda 2
cc2 = 385
cv.circle(img, (cc2, 310), 45, (0, 0, 0), -1) #LLanta
cv.circle(img, (cc2, 310), 25, (100, 100, 100), -1) #Rin exterior
cv.circle(img, (cc2, 310), 10, (150, 150, 150), -1) #Rin interior

#Gasolina
cv.rectangle(img, (65,235),(85,255),(0,0,0),2)

#Cubrir foco
cv.line(img, (470,320), (490,245), (color_fondo, color_fondo, color_fondo), 15)
cv.line(img, (480,235), (440,220), (color_fondo, color_fondo, color_fondo), 15)
cv.circle(img, (485,230), 15, (color_fondo, color_fondo, color_fondo), -1)

cv.imshow("Carro shido xd",img)
cv.waitKey(0)
cv.destroyAllWindows()