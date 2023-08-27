# Persian handwriting recognition

In this repository, we tried to recognize the Persian handwriting, both letters and digits. Now let's see how it works!

It takes a form like "Test.jpg" as input (There is an empty form in "Form_A5.pdf"), extracts each character using computer vision methods and classifies them using convolutional nueral network.

# Developers

This project has been developed by <a href="https://github.com/sadafnazari97">Sadaf Nazari</a> and <a href="https://github.com/shakib1377">Shakib Karami</a> as the final project of <a href="https://wp.kntu.ac.ir/nasihatkon/teaching/cvug/s2020/">Foundations of Computer Vision Course</a>.

# Brief guide

First you should gather a dataset from Persian handwriting(<a href="https://wp.kntu.ac.ir/nasihatkon/teaching/cvug/s2020/assets/files/project/Persian-digits-and-letter-raw.zip">our raw dataset</a>). Data set contains lots of forms like "Dataset_sample2.jpg" and "Dataset_sample1.jpg".
There is an empty sample as "Dataset_Form_A5.pdf".

As the first step, we should extract data from our dataset.

gathering_data_from_dataset.py :

Assume that we have a "dataset" folder, which contains our forms. 
Each image's name follows a particular pattern(like "9999999.jpg").
we read the image, apply perspective transform with detecting the location of arucos.
Then find edges with Canny edge detector which helps us to reduce the effect of noise.
Then we apply Harris corner detection. Guess the location of main corners(top lef ones) and find the nearest corner of those points, extract each cell and save them in "extraced_dataset" (<a href="https://wp.kntu.ac.ir/nasihatkon/teaching/cvug/s2020/assets/files/project/Persian-digits-and-letters-extracted.zip">our extracted dataset</a>)


train_letters.py :

We designed the neural network which classifies letters into 33 classes.(alef-y and blank) and extract "letters.h5" from that.

train_digits.py : 

We designed the neural network which classifies digits into 11 classes.(0-9 and blank) and extract "numbers.h5" from that.


gathering_data_from_form.py : 

Same approach implemented here, output will be saved on a folder which has the equal name of the input's name.(Here if we give "Test.jpg" to it, our file would be "test")

prediction.py : 

Finally we use weights for classifying inputs!
