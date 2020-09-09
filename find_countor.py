import cv2 as cv2

def getEdgeImage(image):
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find Canny edges
    _, binary = cv2.threshold(gray, 40, 50, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # draw all contours
    #image = cv2.drawContours(image, contours, -1, (0, 255, 0), 5, cv2.LINE_AA)
    return contours


