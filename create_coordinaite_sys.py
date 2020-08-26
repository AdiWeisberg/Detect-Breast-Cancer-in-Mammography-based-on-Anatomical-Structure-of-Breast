import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

import find_borders
import math
import imutils

"""
1. finding width of the coordinaite:
"""
#Find an equation straight to the muscle line you tagged
#by the first and the last point we found in find_borders
#return line object
def Finding_Equation_Line(point1, point2):
    point1 = np.array(point1)
    print(point1)
    point2 = np.array(point2)
    print(point2)
    z = np.polyfit(point1, point2, 1)
    m, b = z
    print(m, b)
    line_length = np.poly1d(z)
    plt.plot(point1, m*point1 + b)
    plt.show()
    return line_length



# finding the normal of the muscle line for the nipple line.
# find the the slope of the line of the muscale(m1) and then calculate the normal:
# m2 = -1 / m1.
# return the normal number.
def Finding_Normal(line, point1 = [1,2]):
    x = np.array(point1)
    m, b = line
    m_normal = -1 / m
    print(m_normal)
    plt.plot(x, m_normal * x + b)
    plt.show()
    return m_normal

# for the nipple line:
# circle - get the circle ( the circle taf of the nipple )
# slope - the number we get from the function "finding normal" that calculates the slope
# y-y1 = m2(x-x1)
# return line
def Finding_Equation_Line_By_Slope_And_Point(slope):
    x, y, r = center_point
    m = slope
    b = y - slope * x
    line_width = np.poly1d((m, b))
    return line_width


#by Intersection Point betweeen line nipple and line muscle.
#return width number
# x = (b2 - b1) / (m1 - m2)
# put x in some equation and got y
def Find_intercept_width_length(line_length, line_width):
    m1, b1 = line_length
    m2, b2 = line_width
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return (x, y)

# This function gets 2 points of the nipple and the iterception point with the
# length and width equation lines.
# the function returns the width in float.
def Find_Width(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    # Calculating distance
    return math.sqrt(math.pow(x2 - x1, 2) +
                     math.pow(y2 - y1, 2) * 1.0)


"""
2. finding Length
"""
#we have all the point on the breast countor by the code in- find_countor and we will find the parabola by
#Three or more central points as far apart as possible.
#return parabola equation
#TODO BY NAOMI
def Finding_Prabola_By_Countor():
    pass

#by Intersection Point betweeen Parabola and LineMuscle
#return two point -first and last on the muscle.
#TODO BY: NAOMI
def Finding_Inter_Point_Between_Parabola_LineMuscle(Parabola, LineMuscle):
    pass
#TODO BY: NAOMI
#return the length of the full muscla line.
def Finding_Length(pointFirst, pointlast):
    pass

"""
3. run on all the images:
big for that run on all the images and calculate the length and the width and store them in  two arrays.
and then calculate for each array its AVG.
"""
def calculate_Lengths_and_widths_avg():
    pass

"""
3. Image rotation
"""
def angle_calc(line):
    m1, b = line
    m2 = 1
    return math.degrees(math.atan(np.absolute((m2 - m1)/(1+m2*m1))))

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    cv2.namedWindow('output2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output2', 600, 600)
    cv2.imshow("output2", image)
    cv2.waitKey(0)
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    #cv2.warpAffine(image, M, (nW, nH))
    cv2.namedWindow('output3', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output3', 600, 600)
    cv2.imshow("output3", cv2.warpAffine(image, M, (nW, nH)))
    cv2.waitKey(0)


# Read an image
image = cv2.imread("1-1.png")
line_detected = np.copy(image)
top_line, buttom_line = find_borders.findLine(image)
cv2.line(line_detected, top_line, buttom_line, (208, 216, 75), 15)  # draw and show line between 2 points that we found
output, circle, center_point = find_borders.fineCircle(image)
print(circle)



#################################
eq_line_length = Finding_Equation_Line(top_line, buttom_line)
normal = Finding_Normal(eq_line_length)
eq_line_width = Finding_Equation_Line_By_Slope_And_Point(normal)
intercept_width_length = Find_intercept_width_length(eq_line_length, eq_line_width)

angle = angle_calc(eq_line_length)
print("angle = ", angle)
rotated = ndimage.rotate(image, angle)
#rotate_bound(image, angle)

# view images:
cv2.namedWindow('output1', cv2.WINDOW_NORMAL)
cv2.resizeWindow('output1', 600, 600)
cv2.imshow("output1", rotated)
cv2.waitKey(0)
