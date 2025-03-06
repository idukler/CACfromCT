import os
import torch
from torch import torchvision
import pandas as pd 
from torchvision.io import read_image
import pydicom

# Extract the labels here
PATH_TO_NUMERICAL="data/numerical_data.csv"

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
    
    def process_dicom_file(self):
        for patient_type in os.listdir(self.dicom_dir):

            patient_type_dir = os.path.join(self.dicom_dir, patient_type)
            #
            for patient in os.listdir(patient_type_dir):
                
                patient_data_dir = os.path.join(patient_type_dir, patient, 'SE0001')

                # Read all DICOM files in the directory
                for curr_slice_name in os.listdir(patient_data_dir):

                    # add the current patient's slices
                    slice = pydicom.dcmread(os.path.join(patient_data_dir, curr_slice_name))
                    image = Image.fromarray(slice.pixel_array)   

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