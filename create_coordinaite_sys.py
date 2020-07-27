import numpy
"""
1. finding width of the coordinaite:
"""
#Find an equation straight to the muscle line you tagged
#by the first and the last point we finde in find_borfers
#return line object
#TODO BY: ADI
def Finding_Equation_Line(point1, point2):
    pass

#finding the normal of the muscle line for the nipple line.
#finde the Straight slope and by that the nomal.
#return the normal number.
#TODO BY: MICHAL
def Finding_Normal(line):
    pass
#for the nipple line:
#return line
#TODO BY: MICHAL
def Finding_Equation_Line_By_Slope_And_Point(slope,point):
    pass


#by Intersection Point betweeen line nipple and line muscle.
#return width number
#TODO BY: ADI
def Finding_width(lineNipple,lineMuscle):
    pass

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
