import os

in_root_dir = "data/Contours/"
out_root_dir = "data/images/"

removed_count = 0

#remove contours with no images
for subject_folder in os.listdir(in_root_dir):
    for class_folder in os.listdir(os.path.join(in_root_dir,subject_folder)):
        if class_folder[0] == '.':
            continue
        for contourfile in os.listdir(os.path.join(in_root_dir,subject_folder,class_folder)):
            if not os.path.isfile(os.path.join(out_root_dir,subject_folder,class_folder,contourfile.replace(".txt",".jpg"))):
                print("Removing: " + os.path.join(in_root_dir,subject_folder,class_folder,contourfile))
                os.remove(os.path.join(in_root_dir,subject_folder,class_folder,contourfile))
                removed_count += 1

print("Removed " + str(removed_count) + " contours")