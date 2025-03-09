import os
import torch
import pandas as pd 
from torchvision.io import read_image
import pydicom

# Extract the labels here
PATH_TO_NUMERICAL="numerical_data_reordered.xlsx"

def get_labels_from_numerical_dataset():
    # Get df
    numerical_df = pd.read_excel(PATH_TO_NUMERICAL)
    numerical_dataset = numerical_df.iloc[1:119]

    # Get the labels
    numerical_dataset_labels = numerical_dataset['Calcium score']
    numerical_dataset_labels = numerical_dataset_labels.to_numpy()
    numerical_dataset_labels = numerical_dataset_labels.astype('float32')
    return numerical_dataset_labels


class Artery_CT_Dataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, dicom_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.dicom_dir = dicom_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, patient_type, idx):
        # For healthy patients 
        if patient_type == "Healthy":
            patient_path = os.path.join(self.img_dir, "Healthy")
            for patient in os.listdir(patient_path):
                img_path = os.path.join(self.img_dir, patient, self.img_labels.iloc[idx, 0])
                
                
            img_path = os.path.join(self.img_dir, "Healthy", self.img_labels.iloc[idx, 0])
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    
    if __name__ == "__main__":
        numerical_dataset_labels = get_labels_from_numerical_dataset()
        print(numerical_dataset_labels)
        
        
        
# class Artery_CT_Dataset(torch.utils.data.Dataset):
#     def __init__(self, annotations_file, dicom_dir, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.dicom_dir = dicom_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.dicom_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label

#     if __name__ == "__main__":
#         numerical_dataset_labels = get_labels_from_numerical_dataset()
#         print(numerical_dataset_labels)