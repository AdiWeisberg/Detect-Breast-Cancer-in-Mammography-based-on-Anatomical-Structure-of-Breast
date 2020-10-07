import pydicom as dicom
import os
import cv2
import shutil
import random
from random import shuffle
from math import floor

start_path = "...\\TCIABreast"
ctrain_path = start_path+"\\Calc-Training"
mtrain_path = start_path+"\\Mass-Training\\CBIS-DDSM"
mtest_path = start_path+"\\Mass-Test"
ctest_path = start_path+"\\Calc-Test\\CBIS-DDSM"
path_list = [("Calc-Training", ctrain_path, "Train", "Calc"), ("Mass-Training", mtrain_path, "Train", "Mass"),
             ("Mass-Test", mtest_path, "Test", "Mass"), ("Calc-Test", ctest_path, "Test", "Calc")]
images_path = os.listdir(start_path)

# Specify the output folder3
result_path = "E:\\desktop\\Detect-Breast-Cancer-in-Mammography-based-on-Anatomical-Structure-of-Breast\\breast_dataset"
images_path = os.listdir(start_path)
os.makedirs(result_path + "\\" + "Train" + "\\" +"Mass")
os.makedirs(result_path + "\\" + "Train" + "\\" +"Calc")
os.makedirs(result_path + "\\" + "Test" + "\\" +"Mass")
os.makedirs(result_path + "\\" + "Test" + "\\" +"Calc")
os.makedirs(result_path + "\\" + "Val" + "\\" +"Calc")
os.makedirs(result_path + "\\" + "Val" + "\\" +"Mass")

#for split the train to 70,30 for the val:
train_mass = result_path + "\\" + "Train" + "\\" +"Mass"
train_calc = result_path + "\\" + "Train" + "\\" +"Calc"
val_mass = result_path + "\\" + "Val" + "\\" +"Mass"
val_calc = result_path + "\\" + "Val" + "\\" +"Calc"

for folder_name, path, flag1, flag2 in path_list:
    all_images = os.listdir(path)
    print("path_images = ", all_images)

    for n, image_name in enumerate(sorted(all_images)):
        print(folder_name, image_name)
        sub1 = os.listdir(path + "\\" + image_name)
        sub2 = os.listdir(path + "\\" + image_name + "\\" + sub1[0])
        full_path = os.path.join(path, image_name, sub1[0], sub2[0])
        files_in_folder = os.listdir(full_path)
        # if(len(files_in_folder) != 2):
        #     raise Exception('there is not png image in directory: {} '.format(full_path))
        # # Specify the output folder path that will contain 2 files - source and tagged image.
        new_location = result_path + "\\" + flag1 + "\\" + flag2 + "\\" + image_name
        os.makedirs(new_location)
        # convert dcm to png:
        dcm_to_png_file = full_path + "\\1-1.dcm"
        ds = dicom.dcmread(dcm_to_png_file)
        pixel_array_numpy = ds.pixel_array
        save_path = new_location + "\\" + "Source.png"
        print("dcm to png for file " + str(n) + " succeeded? ", str(cv2.imwrite(save_path, pixel_array_numpy)))
        for filename in files_in_folder:
            if filename.endswith(".png"):
                png_new_location = new_location + "\\" + "Tagged.png"
                png_old_location = full_path + "\\" + "1-1.png"
                shutil.move(png_old_location, png_new_location)



def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda file: file.endswith('.png'), all_files))
    return data_files

#split the data from train to val:

mass_train_list = get_file_list_from_dir(train_mass)#list of all the images of mass train
calc_train_list = get_file_list_from_dir(train_calc)#list of all the images of calc train
len_mass = floor(mass_train_list.__len__()*0.3)

len_calc = floor(calc_train_list.__len__()*0.3)

i = 0
for im in mass_train_list:
    if i <= len_mass:
        i = i+1
        shutil.move(train_mass + "\\"+im, val_mass + "\\"+im)
i = 0
for im in calc_train_list:
    if i <= len_calc:
        i = i + 1
        shutil.move(train_calc + "\\" + im, val_calc + "\\" + im)

