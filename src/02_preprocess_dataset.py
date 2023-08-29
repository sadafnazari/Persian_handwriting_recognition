import os

def make_directories(path):



def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    splited_dataset_path = os.path.join(data_dir, "final")

    make_directories_dataset_split()

if __name__ == "__main__":
    main()
