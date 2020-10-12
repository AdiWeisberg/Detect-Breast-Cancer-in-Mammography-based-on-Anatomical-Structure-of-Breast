import os
import pydicom as dicom
import cv2


def dcm_to_png(full_path):
    try:
        dcm_to_png_file = full_path + "\\" + "1-1.dcm"
        ds = dicom.dcmread(dcm_to_png_file)
        pixel_array_numpy = ds.pixel_array
        save_path = full_path +"\\" + "source.png"
        print("dcm to png for file succeeded? ", str(cv2.imwrite(save_path, pixel_array_numpy)))
        
        # Delete 1-1.dcm file 
        del_file = full_path +"\\" + "1-1.dcm"
        if os.path.exists(del_file):
            os.remove(del_file)

    except OSError as e:
        print("error!")


def is_flip_to_left(png_image):
    """
Equates the threshing of each side in the image.
If the sum of color on the right side is higher on the left side it means
 there is more white there, so the breast is on the right side
    :param img:cv2 image
    :return:boolean- true if the breast is in the right side.
    """
    height, width = png_image.shape[:2]
    width_half = width // 2

    ret, thresh_img = cv2.threshold(png_image, 30, 255, cv2.THRESH_BINARY)
    right = (thresh_img[0:height, width_half:width] // 250).sum()
    left = (thresh_img[0:height, 0:width_half] // 250).sum()

    if left < right:
        return True
    else:
        return False
