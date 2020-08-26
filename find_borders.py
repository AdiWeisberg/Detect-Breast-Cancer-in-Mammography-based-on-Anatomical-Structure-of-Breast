import cv2
import numpy as np
from operator import itemgetter

global top_line, buttom_line, center_point

# ------------- Function to extract blue line by it's color ------------- #
def extract_obj(img, lower,upper):
    HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Get only blue region
    mask = cv2.inRange(HSV_img, lower, upper)
    # Bitwise-AND use mask to extract color region
    obj = cv2.bitwise_and(img,img, mask= mask)
    return obj

def findLine(image):
    draw_lines = np.copy(image)
    # Blue line mask
    # define range of blue color in HSV
    muscle_line = extract_obj(image, np.array([110, 50, 50]), np.array([130,255,255]))
    # blur with Gaussian filter to reduce noise
    blurred_line = cv2.GaussianBlur(muscle_line, (7, 7), 0)
    # edges detection
    edges = cv2.Canny(blurred_line, 20, 255)
    #cv2.namedWindow('Canny', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Canny', 600, 600)
    #cv2.imshow('Canny', edges)
    #cv2.waitKey()
    points = list()
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 25, maxLineGap=200)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(draw_lines, (x1, y1), (x2, y2), (0, 0, 255), 10)  # draw and show line between 2 points that we found
            print("x1: ", x1 ,"y1: ", y1,"x2: ", x2,"y2: ",y2)
            points.append((x1, y1))
            points.append((x2, y2))

    #cv2.namedWindow('draw_lines', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('draw_lines', 600, 600)
    #cv2.imshow('draw_lines', draw_lines)
    #cv2.waitKey()
    # using min() and max()
    # to get min and max in list of tuples
    bottom = (0, 0)
    top = (0, 0)
    for item in points:
        top = min(points, key=itemgetter(1))
        bottom = max(points, key=itemgetter(1))
    print("top -> ", top, "Bottom -> ", bottom)
    return top, bottom


def fineCircle(image):
    """
	function that find the circle in the image that represent the niple.
	use cv2.HoughCircles with 1.0 to the size of the circle.
	:return output:cv2.circle a object of circle.
			circles: tuple of  (x,y) the center of the circle and r the radius.
	"""

    output = image.copy()
    muscle = extract_obj(output, np.array([0,50,50]),np.array([10,255,255]))
    blurred_line = cv2.GaussianBlur(muscle, (7, 7), 0)
    # edges detection
    edges = cv2.Canny(blurred_line, 20, 255)
    #cv2.namedWindow('Canny', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Canny', 600, 600)
    #cv2.imshow('Canny', edges)
    #cv2.waitKey()
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1.3, 100)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        x_sum = y_sum = r_sum = 0
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 10)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            x_sum += x
            y_sum += y
            r_sum += r
            # show the output image
    x_avg = x_sum / len(circles)
    y_avg = y_sum / len(circles)
    r_avg = r_sum / len(circles)
    return output, circles, (x_avg, y_avg, r_avg)

# -------------------------- main -------------------------- #
if __name__ == '__main__':
    # Read an image
    image = cv2.imread("1-1.png")
    line_detected = np.copy(image)
    top_line, buttom_line = findLine(image)
    #cv2.line(line_detected, top_line, buttom_line, (208, 216, 75), 15)  # draw and show line between 2 points that we found
    #cv2.namedWindow('line_detected', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('line_detected', 600, 600)
    #cv2.imshow('line_detected', line_detected)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    output, circle, center_point = fineCircle(image)
    print(circle)
    #cv2.namedWindow('output1', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('output1', 600, 600)
    #cv2.imshow("output1",output)
    #cv2.waitKey(0)
