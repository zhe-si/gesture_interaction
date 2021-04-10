# 生成标签csv
import os


def main():
    file_path = r"F:\My_Resources\datasets\GestureRecognition_fwwb\dataset\jester-v1-labels.csv"
    labels_path = r"F:\My_Resources\datasets\GestureRecognition_fwwb\video"
    labels = os.listdir(labels_path)
    with open(file_path, "w") as label_csv:
        for label in labels:
            label_csv.write(label + "\n")
    pass


if __name__ == '__main__':
    main()
