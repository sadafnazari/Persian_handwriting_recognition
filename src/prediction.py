import glob
import os

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from utils.config import check_config_file, check_config_keys
from utils.model import build_model
from utils.preprocessing import preprocess


def predict(config):

    required_keys = [
        "test_forms.extracted_path",
        "pre_processing.cell_width",
        "pre_processing.cell_height",
        "pre_processing.num_classes",
        "pre_processing.gaussian_kernel",
        "model.model_numbers_path",
        "model.model_letters_path",
        "inference.threshold",
    ]
    check_config_keys(config, required_keys)

    form_path = config["test_forms"].get("extracted_path")

    cell_width = config["pre_processing"].get("cell_width")
    cell_height = config["pre_processing"].get("cell_height")

    cell_width = config["pre_processing"].get("cell_width")
    cell_height = config["pre_processing"].get("cell_height")

    gaussian_kernel = config["pre_processing"].get("gaussian_kernel")

    num_classes_numbers = config["pre_processing"].get("num_classes_numbers")
    num_classes_letters = config["pre_processing"].get("num_classes_letters")

    model_numbers_path = config["model"].get("model_numbers_path")
    model_letters_path = config["model"].get("model_letters_path")

    threshold = config["inference"].get("threshold")

    model_numbers = build_model(num_classes_numbers, cell_width, cell_height)
    model_numbers.load_weights(model_numbers_path)

    model_letters = build_model(num_classes_letters, cell_width, cell_height)
    model_letters.load_weights(model_letters_path)

    classes_numbers = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9"
    ]
    classes_letters = [
        "ا",
        "ب",
        "پ",
        "ت",
        "ث",
        "ج",
        "چ",
        "ح",
        "خ",
        "د",
        "ذ",
        "ر",
        "ز",
        "ژ",
        "س",
        "ش",
        "ص",
        "ض",
        "ط",
        "ظ",
        "ع",
        "غ",
        "ف",
        "ق",
        "ک",
        "گ",
        "ل",
        "م",
        "ن",
        "و",
        "ه",
        "ی",
    ]

    intensity_bs = 0
    intensity_ms = 0
    intensity_phd = 0

    first_name = ""
    last_name = ""
    student_id = ""

    forms = glob.glob(form_path + "/*")
    for data_path in forms:
        test_path = sorted(glob.glob(data_path + "/*.jpg"))
        for path in test_path:
            image_name = os.path.basename(path)
            # image = load_img(path, target_size=(cell_width, cell_height))
            image = cv2.imread(path)
            image = cv2.resize(image, (cell_width, cell_height))
            image = preprocess(image, gaussian_kernel)
            image = img_to_array(image) / 255.0
            image = np.expand_dims(image, axis=0)
            if "ID" in image_name:
                predicted = model_numbers.predict(image)[0]
                if predicted[np.argmax(predicted)] >=  threshold:
                    student_id += str(classes_numbers[np.argmax(predicted)])
                else:
                    student_id += " "
            elif "FN" in image_name:
                predicted = model_letters.predict(image)[0]
                if predicted[np.argmax(predicted)] >=  threshold:
                    first_name += str(classes_letters[np.argmax(predicted)]) + " "
                else:
                    first_name += " "
            elif "LN" in image_name:
                predicted = model_letters.predict(image)[0]
                if predicted[np.argmax(predicted)] >=  threshold:
                    last_name += str(classes_letters[np.argmax(predicted)]) + " "
                else:
                    last_name += " "
            elif "BS" in image_name:
                for i in range(cell_width):
                    for j in range(cell_height):
                        intensity_bs += (
                            image[0][i][j]
                        )
            elif "MS" in image_name:
                for i in range(cell_width):
                    for j in range(cell_height):
                        intensity_ms += (
                            image[0][i][j]
                        )
            elif "PHD" in image_name:
                for i in range(cell_width):
                    for j in range(cell_height):
                        intensity_phd += (
                            image[0][i][j]
                        )

        form_name = os.path.basename(data_path)
        print("Form: ", form_name)
        print("Student ID: ", student_id)
        print("First name: ", first_name)
        print("Last name: ", last_name)
        field = max(intensity_bs, intensity_ms, intensity_phd)
        if field == intensity_bs:
            print("field: ", "کارشناسی")
        elif field == intensity_ms:
            print("field: ", "کارشناسی ارشد")
        else:
            print("field: ", "دکتری")


if __name__ == "__main__":
    config_path = "config/config.yaml"
    config = check_config_file(config_path)
    predict(config)
