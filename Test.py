import cv2
import numpy as np

# Crear imagen negra con dos manchas de diferente tamaño
img = np.zeros((300,300), dtype=np.uint8)
cv2.circle(img, (80,150), 40, 255, -1)   # Mancha grande
cv2.circle(img, (200,150), 20, 255, -1)  # Mancha pequeña

# Obtener contornos
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = max(contours, key=cv2.contourArea)   # Tomamos el mayor contorno

# ================================
# Centroide con momentos
M = cv2.moments(c)
cx_m = int(M["m10"] / M["m00"])
cy_m = int(M["m01"] / M["m00"])

# ================================
# Centro geométrico con bounding box
x, y, w, h = cv2.boundingRect(c)
cx_b = x + w//2
cy_b = y + h//2

# ================================
# Dibujar resultados
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_color, [c], -1, (0,255,0), 2)           # Contorno en verde
cv2.circle(img_color, (cx_m, cy_m), 5, (0,0,255), -1)        # Centroide en rojo
cv2.circle(img_color, (cx_b, cy_b), 5, (255,0,0), -1)        # Bounding box en azul
cv2.rectangle(img_color, (x,y), (x+w,y+h), (255,255,0), 2)   # Caja delimitadora

cv2.imshow("Centroide (rojo) vs BoundingBox (azul)", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Centroide (momentos):", (cx_m, cy_m))
print("Centro bounding box:", (cx_b, cy_b))