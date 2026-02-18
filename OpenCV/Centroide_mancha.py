import cv2
import numpy as np

video = cv2.VideoCapture(0)

while video.isOpened():
    ret, vid = video.read()
    if not ret:
        break

    #Para detectar colores mas efectivamente
    frame_hsv = cv2.cvtColor(vid, cv2.COLOR_BGR2HSV)

    # Rango de azul (hay dos rangos en HSV)
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(frame_hsv, lower_blue, upper_blue)

    result = cv2.bitwise_and(vid, vid, mask=mask)

    # 1 solo canal
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Obtener contornos
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    try:
        c = max(contours, key=cv2.contourArea)  # Tomamos el mayor contorno
    except ValueError:
        print(">>> No color detected ...")
        if cv2.waitKey(350) & 0xFF == ord('q'):
            break
        continue

    # ================================
    # Centroide con momentos
    M = cv2.moments(c)
    try:
        cx_m = int(M["m10"] / M["m00"])
        cy_m = int(M["m01"] / M["m00"])
    except ZeroDivisionError:
        continue

    # ================================
    # Centro geométrico con bounding box
    x, y, w, h = cv2.boundingRect(c)
    cx_b = x + w // 2
    cy_b = y + h // 2

    # ================================
    # Dibujar resultados
    img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_color, [c], -1, (0, 255, 0), 2)  # Contorno en verde
    cv2.circle(img_color, (cx_m, cy_m), 5, (0, 0, 255), -1)  # Centroide en rojo
    cv2.circle(img_color, (cx_b, cy_b), 5, (255, 0, 0), -1)  # Bounding box en azul
    cv2.rectangle(img_color, (x, y), (x + w, y + h), (255, 255, 0), 2)

    cv2.imshow("normal",vid)
    cv2.imshow("Only blue", result)
    cv2.imshow("Final", img_color)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()