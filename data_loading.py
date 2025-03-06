import pydicom
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch


# Directory containing your DICOM files
dicom_dir="dataset_full"
output_dir="dataset_full_images"
tensor_dir="dataset_full_tensors"


def dsm_to_tensor():
    # Iterate through each patient type
    # healthy or patient
    for patient_type in os.listdir(dicom_dir):
        # # testing small to see if this works
        # if patient_type == "Patient":
        #     continue

        patient_type_dir = os.path.join(dicom_dir, patient_type)
        
        for patient in os.listdir(patient_type_dir):
            
            # # testing small to see if this works
            # if patient != "Patient_1":
            #     continue
            
            patient_data_dir = os.path.join(patient_type_dir, patient, 'SE0001')

            # Read all DICOM files in the directory            
            slices = []
            for curr_slice_name in os.listdir(patient_data_dir):
                                

                # add the current patient's slices
                slice = pydicom.dcmread(os.path.join(patient_data_dir, curr_slice_name))
                pixel_array = slice.pixel_array
                
                # Normalize the pixel array to the range [0, 255]
                pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255
                pixel_array = pixel_array.astype(np.uint8)
                
                # Convert the NumPy array to a PyTorch tensor
                tensor = torch.tensor(pixel_array, dtype=torch.uint8)
                                
                slices.append(tensor)
                
               
            # Create a tensor for the patient
            patient_tensor = torch.stack(slices, dim=0)
            # Saving slice to dataset_full_images
            tensor_path = f"{patient_type}_{patient}.pt"
            path_to_save_image_slice = os.path.join(tensor_dir, tensor_path)
            
            # Create direcotirs if they dont exist
            os.makedirs(os.path.dirname(path_to_save_image_slice), exist_ok=True)

            # Save the tensor to a file
            torch.save(patient_tensor, path_to_save_image_slice)
            
# works for the trivial example
def load_tensor_test(path):
    tensor = torch.load(path)
    print(tensor.shape)
    print(tensor.dtype)
    print(tensor.device)
    print(tensor)
    
    # for i in range(tensor.shape[0]):
    plt.imshow(tensor[0], cmap='gray')
    plt.show()


def dsm_to_png():
    # Iterate through each patient type
    # healthy or patient
    for patient_type in os.listdir(dicom_dir):
        # testing small to see if this works
        if patient_type == "Patient":
            continue

        patient_type_dir = os.path.join(dicom_dir, patient_type)
        
        for patient in os.listdir(patient_type_dir):
            
            # testing small to see if this works
            if patient != "Patient_1":
                continue
            
            patient_data_dir = os.path.join(patient_type_dir, patient, 'SE0001')

            # Read all DICOM files in the directory
            for curr_slice_name in os.listdir(patient_data_dir):
                

                # add the current patient's slices
                slice = pydicom.dcmread(os.path.join(patient_data_dir, curr_slice_name))
                pixel_array = slice.pixel_array
                
                # Normalize the pixel array to the range [0, 255]
                pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255
                pixel_array = pixel_array.astype(np.uint8)
                
                image = Image.fromarray(pixel_array)
                
                # plt.imshow(image, cmap='gray')
                # plt.show()
                            
                base_name = os.path.basename(curr_slice_name)
                curr_slice_name = os.path.splitext(base_name)[0]

                # Saving slice to dataset_full_images
                image_path = f"{curr_slice_name}.png"
                path_to_save_image_slice = os.path.join(output_dir, patient_type, patient, image_path)
                
                # Create direcotirs if they dont exist
                os.makedirs(os.path.dirname(path_to_save_image_slice), exist_ok=True)

                
                # Save image at path
                image.save(path_to_save_image_slice)

# okay min number of slices is 92
def min_num_slices():
    min_num_slices = float('inf') # big number so things are smaller than it
    
    for patient_type in os.listdir(output_dir):

        patient_type_dir = os.path.join(output_dir, patient_type)
        
        # check for each patient the number of slices they have
        for patient in os.listdir(patient_type_dir):
            
            patient_data_dir = os.path.join(patient_type_dir, patient)

            curr_num_slices = len(os.listdir(patient_data_dir))
            if curr_num_slices < min_num_slices:
                min_num_slices = curr_num_slices
                
                
    return min_num_slices

if __name__ == "__main__":
    # dsm_to_tensor()
    load_tensor_test("/Users/ido_dukler/Documents/UCLAYear1/BIOE228/Project/dataset_full_tensors/Healthy_Patient_1.pt")