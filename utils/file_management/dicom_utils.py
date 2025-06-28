import glob
import os
import pydicom
import gzip
import numpy as np

'''
Add functions for all things dicoms here so they can be modified for each project. 

Terminology:
dir = directory
file = file
path = full path from root (ex. /data/mochila1/.../patient/date/series)
name = current file or directory without the full path
'''

def is_dicom_file(filepath):
    """
    Check if the file at the given path is a DICOM file.
    """
    fname = os.path.split(filepath)[-1]
    if not os.path.isdir(filepath) and not fname.startswith('._') and filepath.lower().endswith(('.dcm', '.dcm.gz')):
        return True
    return False

def get_dicom_paths(directory, search_method='recursive'):
    """
    Get a list of all DICOM file paths in a directory using the specified search method.
    """
    assert search_method in ['recursive', 'glob'], "Invalid search method. Expect 'recursive' or 'glob'."

    if search_method == 'glob':
        filepaths = glob.glob(directory+'/*.DCM.gz') + glob.glob(directory+'/*.DCM') + glob.glob(directory+'/*.dcm')
    else:  # recursive search
        filepaths = [os.path.join(directory, f) for f in os.listdir(directory) if is_dicom_file(os.path.join(directory, f))]

    return filepaths

def read_dicom(filepath:str):
    """
    Read a DICOM file.
    """
    assert is_dicom_file(filepath), "invalid file path"

    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rb') as f:
            ds = pydicom.dcmread(filepath)
    else:
        ds = pydicom.dcmread(filepath)
    return ds

def read_all_dicoms(directory, sort_by='InstanceNumber'):
    """
    Read all DICOM files in a directory, optionally decompress them, and sort by the specified metadata field.
    """
    dicom_paths = get_dicom_paths(directory, search_method='recursive')
    dicoms = [read_dicom(path) for path in dicom_paths]

    # Sort DICOMs by the specified metadata field
    dicoms.sort(key=lambda ds: float(getattr(ds, sort_by, float('inf'))))

    return dicoms

def read_one_dicom(directory, index:int=0):
    """
    Read the first valid DICOM file found in a directory.
    """
    dicom_paths = get_dicom_paths(directory, search_method='recursive')
    if dicom_paths:
        return read_dicom(dicom_paths[index])
    return None

def get_volume_from_directory(directory, sort_by_metadata_field='InstanceNumber'):
    """
    Read DICOM files from a directory and return the 3D volume and the first DICOM dataset.
    Params:

    Returns:
    - volume (numpy.ndarray): 3D volume constructed from the DICOM pixel data.
    - first_dataset (pydicom.dataset.FileDataset): The first DICOM metadata in the sorted list.
    """
    ds_list = read_all_dicoms(directory, sort_by_metadata_field)
    
    if not ds_list:
        raise FileNotFoundError("No DICOM files found in the directory.")
    
    volume = np.array([ds.pixel_array for ds in ds_list])
    
    return volume, ds_list[0]

# def dcm_read_dir(directory, decompress=False, temp_dir=''):
#     # filenames = glob.glob(directory + '/*.DCM') + glob.glob(directory + '/*.dcm')
#     # Handle missing file extensions from some institutions
#     filenames = []
#     for f in os.listdir(directory):
#         if not os.path.isdir(os.path.join(directory, f)) and not f.startswith('._'):
#             try:
#                 dcm = pydicom.read_file(os.path.join(directory, f)) # if can be opened, then dicom
#                 filenames.append(os.path.join(directory, f))
#             except:
#                 pass

#     # Create list of opened dicoms sorted by instance number
#     instance_map = {}
#     for fn in filenames:
#         if decompress:
#             tempfname = os.path.join(temp_dir, 'temp.DCM')
#             subprocess.run('dcmdjpeg -v {} {}'.format(fn, tempfname), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             ds = pydicom.dcmread(tempfname)
#             os.remove(tempfname)
#         else:
#             ds = pydicom.dcmread(fn)
#         instance_map[int(ds.InstanceNumber)] = ds

#     instance_numbers = list(instance_map.keys())
#     instance_numbers.sort()

#     ds_list = [None]*len(instance_numbers) #is this really necessary
#     for i, instance_num in enumerate(instance_numbers):
#         ds_list[i] = instance_map[instance_num]
#     return ds_list


# def read_dicoms_from_directory(directory, decompress=False, temp_dir='', sort_by_metadata_field:str='InstanceNumber'):
#     '''
#     Read all dicoms in the folder to load the volume. This includes the intensity values and metadata. Sort the dicoms files 
#     by instance number and return in a list.
#     '''

#     dicom_paths = get_all_dicom_paths(directory,search_method='recursive')

#     # Load DICOM files and sort by image position patient
#     ds_list = []
#     for dicom_fpath in dicom_paths:
#         if decompress:
#             tempfname = os.path.join(temp_dir, 'temp.DCM')
#             subprocess.run('dcmdjpeg -v {} {}'.format(dicom_fpath, tempfname), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             ds = pydicom.dcmread(tempfname)
#             os.remove(tempfname)
#         else:
#             ds = pydicom.dcmread(dicom_fpath) #vs read_file???
#         ds_list.append(ds)
    
#     ds_list.sort(key=lambda ds: float(getattr(ds,sort_by_metadata_field)))
#     return ds_list