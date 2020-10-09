import cv2 as cv2
import numpy as np


def getEdgeImage(image):
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find Canny edges
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("binary_treshold.png", binary)
    # view images:
    cv2.namedWindow('threshold', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('threshold', 600, 600)
    cv2.imshow("threshold", binary)
    cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    # draw all contours
    image_with_contours = np.copy(image)
    image_with_contours = cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 5, cv2.LINE_AA)
    cv2.imwrite("image_with_contours.png", image_with_contours)

    return contours
