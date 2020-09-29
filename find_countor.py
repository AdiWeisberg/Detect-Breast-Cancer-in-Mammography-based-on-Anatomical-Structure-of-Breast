import cv2 as cv2
import numpy as np


def getEdgeImage(image):
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find Canny edges
    _, binary = cv2.threshold(gray, 23, 25, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    # draw all contours
    image_with_contours = np.copy(image)
    image_with_contours = cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 5, cv2.LINE_AA)
    cv2.imwrite("image_with_contours_RETR_EXTERNAL.png", image_with_contours)
    # view images:
    cv2.namedWindow('output1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output1', 600, 600)
    cv2.imshow("output1", image_with_contours)
    cv2.waitKey(0)
    return contours


