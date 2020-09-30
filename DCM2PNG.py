import pydicom as dicom
import os
import cv2
import shutil
import random

ctrain_path = "E:\\TCIA_Breast_Test\\Calc-Training"
mtrain_path = "E:\\TCIA_Breast_Test\\Mass-Training\\CBIS-DDSM"
mtest_path = "E:\\TCIA_Breast_Test\\Mass-Test"
ctest_path = "E:\\TCIA_Breast_Test\\Calc-Test\\CBIS-DDSM"
path_list = [("Calc-Training", ctrain_path, "Train", "Calc"), ("Mass-Training", mtrain_path, "Train", "Mass"),
             ("Mass-Test", mtest_path, "Test", "Mass"), ("Calc-Test", ctest_path, "Test", "Calc")]
start_path = "E:\\TCIA_Breast_Test"
images_path = os.listdir(start_path)

# Specify the output folder3
result_path = "E:\\breast_dataset"
images_path = os.listdir(start_path)
os.makedirs(result_path + "\\" + "Train" + "\\" +"Mass")
os.makedirs(result_path + "\\" + "Train" + "\\" +"Calc")
os.makedirs(result_path + "\\" + "Test" + "\\" +"Mass")
os.makedirs(result_path + "\\" + "Test" + "\\" +"Calc")
os.makedirs(result_path + "\\" + "Val" + "\\" +"Calc")
os.makedirs(result_path + "\\" + "Val" + "\\" +"Mass")
# TODO: Ramdomly Sampling from Test (flag1)
# precent2val = 30
# validation = []

for folder_name, path, flag1, flag2 in path_list:
    all_images = os.listdir(path)
    print("path_images = ", all_images)
    # TODO: Ramdomly Sampling from Test (flag1)
    # if flag1 == "Test":
    #     precent = (len(all_images) * 15)/100
    #     validation += random.sample(all_images, frac=precent)
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

# TODO: Ramdomly Sampling from Test (flag1)
#print(" length of validation = ". len(validation))


# Delete all contents of a directory using shutil.rmtree() and  handle exceptions - to delete the original folder.
# try:
#    shutil.rmtree(folder_path)
# except:
#    print('Error while deleting directory')
