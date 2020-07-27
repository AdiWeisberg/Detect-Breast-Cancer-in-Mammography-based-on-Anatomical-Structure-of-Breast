import pydicom as dicom
import os
import cv2
import shutil

import PIL # optional
# make it True if you want in PNG format
PNG = False
# Specify the .dcm folder path
folder_path = "Mass_Test"
result_path = "Mass_Test_combined"
# Specify the output jpg/png folder path
png_folder_path = "Mass_Test"
os.makedirs(result_path+"\\"+'Tagged')
os.makedirs(result_path+"\\"+'Source')
images_path = os.listdir(folder_path)
for n, image_name in enumerate(sorted(images_path)):
    sub1 = os.listdir(folder_path+"\\"+image_name)
    sub2 = os.listdir(folder_path+"\\"+image_name+"\\"+sub1[0])
    dcm_to_png_file = os.path.join(folder_path, image_name, sub1[0], sub2[0]) +"\\1-1.dcm"
    ds = dicom.dcmread(dcm_to_png_file)
    pixel_array_numpy = ds.pixel_array
    save_path = result_path+"\\"+'Source'+"\\" + image_name +".png"
    print("success?: " , str(cv2.imwrite(save_path, pixel_array_numpy)))
    old = os.path.join(folder_path, image_name, sub1[0], sub2[0]) +"\\"+"1-1.png"
    old_location = os.path.join(folder_path, image_name, sub1[0], sub2[0]) +"\\"+"Tagged_"+image_name+".png"
    new_location = result_path+"\\"+'Tagged'+"\\"+"Tagged_"+image_name+".png"
    os.rename(old, old_location)
    shutil.move(old_location, new_location)
    if n % 50 == 0:
        print('{} image converted'.format(n))

# Delete all contents of a directory using shutil.rmtree() and  handle exceptions
try:
   shutil.rmtree(folder_path)
except:
   print('Error while deleting directory')
