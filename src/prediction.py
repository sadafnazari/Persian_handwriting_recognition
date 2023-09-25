import glob
import os

import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from utils.config import check_config_file, check_config_keys
from utils.model import build_model


def predict(config):

    required_keys = [
        "test_forms.extracted_path",
        "pre_processing.cell_width",
        "pre_processing.cell_height",
        "pre_processing.num_classes",
        "model.model_path",
    ]
    check_config_keys(config, required_keys)

    form_path = config["test_forms"].get("extracted_path")

    cell_width = config["pre_processing"].get("cell_width")
    cell_height = config["pre_processing"].get("cell_height")

    cell_width = config["pre_processing"].get("cell_width")
    cell_height = config["pre_processing"].get("cell_height")

    num_classes = config["pre_processing"].get("num_classes")

    model_path = config["model"].get("model_path")

    model = build_model(num_classes, cell_width, cell_height)
    model.load_weights(model_path)

    classes = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
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
        test_path = glob.glob(data_path + "/*.jpg")
        for path in test_path:
            image_name = os.path.basename(path)
            image = load_img(path, target_size=(cell_width, cell_height))
            image = img_to_array(image) / 255.0
            image = np.expand_dims(image, axis=0)
            if "ID" in image_name:
                if str(classes[np.argmax(model.predict(image)[0])]) != "empty":
                    student_id += str(classes[np.argmax(model.predict(image)[0])]) + " "
            elif "FN" in image_name:
                if str(classes[np.argmax(model.predict(image)[0])]) != "empty":
                    first_name += str(classes[np.argmax(model.predict(image)[0])])
            elif "LN" in image_name:
                if str(classes[np.argmax(model.predict(image)[0])]) != "empty":
                    last_name += str(classes[np.argmax(model.predict(image)[0])])
            elif "BS" in image_name:
                for i in range(cell_width):
                    for j in range(cell_height):
                        intensity_bs += (
                            image[0][i][j][0] + image[0][i][j][1] + image[0][i][j][2]
                        )
            elif "MS" in image_name:
                for i in range(cell_width):
                    for j in range(cell_height):
                        intensity_ms += (
                            image[0][i][j][0] + image[0][i][j][1] + image[0][i][j][2]
                        )
            elif "PHD" in image_name:
                for i in range(cell_width):
                    for j in range(cell_height):
                        intensity_phd += (
                            image[0][i][j][0] + image[0][i][j][1] + image[0][i][j][2]
                        )

        form_name = os.path.basename(data_path)
        print("Form: ", form_name)
        print("Student ID: ", student_id)
        print("First name: ", first_name)
        print("Last name: ", last_name)
        field = min(intensity_bs, intensity_ms, intensity_phd)
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
