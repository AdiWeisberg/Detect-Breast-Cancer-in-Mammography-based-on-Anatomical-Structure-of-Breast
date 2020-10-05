import cv2
import shutil
import os


# def main(rootDir, newpathimage):
#     """
#     Go over the path, create new path for the bad image in the path -"newpathimage".
#     check the 3 think:
#     1.resolution.
#     2.shape Image.
#     3.bad Image.
#     :param rootDir:  the path for the data to check. not need "\\" in the end.
#     :param newpathimage:the path for the bad image that hase remove. not need "\\" in the end.
#     :return: logs of the Final result- number of bad image  and List the types of bad pictures.
#     """
#     numofimage =0
#     sumNOTgoodimage = 0
#     sumbadresolution = 0
#     sumbadshape = 0
#     sumbadimage = 0
#     newpathimage = newpathimage + "\\"
#     for (_, dirs, _) in os.walk(rootDir):
#         for dir in dirs:
#             for (dirname, _, files) in os.walk(rootDir + "\\" + dir):
#                 for filename in files:
#                     if filename.endswith('.DCM'):
#                         pathImage = dirname+"\\" + filename
#                         img = cv2.imread(pathImage)
#
#                         if badImage(img):
#                             sumNOTgoodimage += 1
#                             sumbadimage += 1
#
#                             img_flip_ud = cv2.flip(img, 0)
#                             cv2.imwrite('data/dst/lena_cv_flip_ud.jpg', img_flip_ud)
#
#                         numofimage += 1



def badImage(img):
    """
Equates the threshing of each side in the image.
If the sum of color on the right side is higher on the left side it means
 there is more white there, so the breast is on the right side
    :param img:cv2 image
    :return:boolean- true if the breast is in the right side.
    """
    height, width = img.shape[:2]
    width_half = width//2

    ret, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    right = (thresh_img[0:height, width_half:width] // 250).sum()
    left = (thresh_img[0:height, 0:width_half] // 250).sum()

    if left < right:
        return True
    else:
        return False


if __name__ == '__main__':

    pathImage = "1-1_1.png"

    if pathImage.endswith(".png"):#cv2 work only on png and NOT on dcm!

        cv2.namedWindow("input", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
        im = cv2.imread(pathImage)  # Read image
        imS = cv2.resize(im, (960, 540))  # Resize image
        cv2.imshow("input", imS)  # Show image
        cv2.waitKey(0)

        if badImage(im):
            print('yes')
            img_flip_ud = cv2.flip(im, 1)#flip the image that the brest will be in the left side
            cv2.namedWindow("output", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
            #im = cv2.imread(pathImage)  # Read image
            imS = cv2.resize(img_flip_ud, (960, 540))  # Resize image
            cv2.imshow("output", imS)  # Show image
            cv2.waitKey(0)
            cv2.imwrite('1-1_1.png', img_flip_ud)

