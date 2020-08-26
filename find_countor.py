import signal

import cv2 as cv2
import numpy as np
from PIL import ImageOps
from matplotlib import pyplot as plt

def getEdgeImage(image):
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find Canny edges
    #edged = cv2.Canny(gray, 200, 200)
    _, binary = cv2.threshold(gray, 23, 25, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # draw all contours
    image = cv2.drawContours(image, contours, -1, (0, 255, 0), 5, cv2.LINE_AA)
    return contours,image


def imageShowWithWait(edgeImage, image):
    #cv2.namedWindow(edgeImage, cv2.WINDOW_NORMAL)
    #cv2.resizeWindow(edgeImage, 600, 600)
    #cv2.imshow(edgeImage, image)
    #cv2.waitKey()
    imgplot = plt.imshow(image)
    plt.show()

#def cut_annotations(image):
#    border = (0, 200, 0, 200)  # left, up, right, bottom
#    ImageOps.crop(image, border)

def main():
    originalImage = cv2.imread('1-1.png')
    #cut_annotations(originalImage)
    cv2.waitKey(0)
    contours, edgeImage = getEdgeImage(originalImage)
    imageShowWithWait("edgeImage", edgeImage)
    #print(contours)
    #listX, listY = zip(*contours[0])
    #degree = 4
    #weights = np.polyfit(listX, listY, degree)
    #poly_equation = np.poly1d(weights)
    #print(poly_equation)
    #print(approx)
    cv2.namedWindow("originalImage", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("originalImage", 600, 600)
    cv2.imshow("originalImage", edgeImage)
    cv2.waitKey(0)

main()
