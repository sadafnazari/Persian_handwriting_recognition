# Persian_handwriting_recognition


# Data collection
The original dataset is taken from the one that is collected by students in the course, which can be found <a href='https://wp.kntu.ac.ir/nasihatkon/teaching/cvug/s2020/assets/files/project/Persian-digits-and-letter-raw.zip'>in the official website of the course</a>. However, in this repository, some modifications are applied to the dataset and some images that were not collected according to the instructions were removed. We recommend to use this dataset can be found in `data/raw_dataset.zip`. It should be unzip and put in `data/raw` directory.

The dataset contains a set of images, taken from forms that can be found in `assets/Dataset_Form_A5.pdf`. Each form contains 4 Aruco markers and a number of cells which should be filled with persian handwritten numbers and letters. These forms have 2 types:
Type 'a': number '0', '1', the first part of the persian alphabet, '2', and '3'.
Type 'b': number '4', '5', the second part of the persian alphabet, '6', '7, '8', and'9'. 

In case of a need to collect a new dataset, or add more data to the current dataset, the following instructions must be followed:
1. Form should be printed, prefrably in A5 size.
2. The orientation of the arucos should be the same as a sample shown in `assets/Dataset_sample1.jpg`.
3. Each row is dedicated to either a number or a letter. 
4. Each row should be filled with the determined order of the specific type of form. It should be either the same as `assets/Dataset_sample1.jpg` (type a) or `assets/Dataset_sample2.jpg` (type b).
5. Image should be taken in format 'jpg'. While the orientation of the arucos should be fixed, the form it self can be in custome distance and orientation.
6. The image should be stored in the `data/raw` directory with the correct subdirectory that corresponds to the type of form.
7. (preprocessing step): Each form will be separated manually in `data/processed/a` or `data/processed/b` based on their type. 
   
# Data preprocessing
The objective of the data preprocessing is to extract each cell from each form, and store each cell in a folder that represents the class of that cell (0 to 9 for the numbers and 10 to 42 for the letters). 

This can be done with the following steps:
1. Since in the original dataset, the images are stored numrically altogether, it is not easy to store the extracted cells into the correct subfolder that would represent the label/class of that cell with only computer vision tools. Therefore, the manual separation of the forms based on their type (a or b) is needed. The separated dataset can be found `data/processed_dataset.zip`. It should be unzip and stored in `/data/processed` directory.
2. Next the script for extracting the final dataset should be run with the following command `python3 src/extract_dataset.py`:
   1. Make the neccessary directories in the `data/final` directory. These directories will later represent the class of each data.
   2. Reads image
   3. Detects the Aruco markers, and drops the ones that their markers cannot be detected
   4. Applies perspective transformation on the image so that the orientation of the form will be fixed for all images
   5. Resize the images to a determined size in order to have the same size for all the images
   6. Extracts each cell, resizes them to a determined size, and store them to their correspondent folder in the `data/final` directory(labeling process)
   7. (Note): The final data can be found in `data/final_dataset.zip` in case one would like to skip executing th `extract_dataset.py` script.
