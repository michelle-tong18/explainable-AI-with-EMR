import numpy as np
import pydicom
import subprocess
import gzip
import os

import pyrootutils
root = pyrootutils.setup_root(
    search_from=os.path.abspath(''),
    indicator=[".git"],
    pythonpath=True, # add root directory to the PYTHONPATH (helps with imports)
    dotenv=True, # load environment variables from .env if exists in root directory
)
from utils.file_management.dicom_utils import read_dicoms_from_directory
import pdb


def dcm_read_dir(directory, decompress=False, temp_dir=''):
    # filenames = glob.glob(directory + '/*.DCM') + glob.glob(directory + '/*.dcm')
    # Handle missing file extensions from some institutions
    filenames = []
    for f in os.listdir(directory):
        if not os.path.isdir(os.path.join(directory, f)) and not f.startswith('._'):
            try:
                dcm = pydicom.read_file(os.path.join(directory, f)) # if can be opened, then dicom
                filenames.append(os.path.join(directory, f))
            except:
                pass

    # Create list of opened dicoms sorted by instance number
    instance_map = {}
    for fn in filenames:
        if decompress:
            tempfname = os.path.join(temp_dir, 'temp.DCM')
            subprocess.run('dcmdjpeg -v {} {}'.format(fn, tempfname), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ds = pydicom.dcmread(tempfname)
            os.remove(tempfname)
        else:
            ds = pydicom.dcmread(fn)
        try:
            test = ds.pixel_array.shape
            instance_map[int(ds.InstanceNumber)] = ds
        except:
            pass

    instance_numbers = list(instance_map.keys())
    instance_numbers.sort()

    ds_list = [None]*len(instance_numbers) #is this really necessary
    for i, instance_num in enumerate(instance_numbers):
        ds_list[i] = instance_map[instance_num]

    return ds_list
    
def check_if_path_to_dcm(path:str):
    fname = os.path.split(path)[-1] # remove path to file
    if not os.path.isdir(path) \
        and not fname.startswith('._') \
        and (path.lower().endswith('.dcm') or path.lower().endswith('.dcm.gz')):
        return True
    else:
        return False 

def get_one_dicom_path(dirpath:str):
    fpath = ''
    for f in os.listdir(dirpath):
        fpath = os.path.join(dirpath, f)
        if check_if_path_to_dcm(fpath)==True:
            return fpath
    return fpath

def read_dicom_from_filepath(fpath:str):
    if fpath.endswith('.gz'):
        with gzip.open(fpath, 'rb') as f:
            ds = pydicom.dcmread(fpath)

    elif check_if_path_to_dcm(fpath)==True:
        ds = pydicom.dcmread(fpath)
    else:
        assert ('invalid file path')

    return ds

def get_volume_from_directory(directory,decompress=False, temp_dir='', sort_by_metadata_field:str='InstanceNumber'):
    ds_list = read_dicoms_from_directory(directory, decompress, temp_dir, sort_by_metadata_field)
    vol = np.array([ds.pixel_array for ds in ds_list])
    return vol, ds_list[0]

#################################################################
# Authors: Eugene Ozhinsky, Madeline Hess
# Sources: https://git.radiology.ucsf.edu/eozhins2/r2cnn_faster-rcnn_tensorflow/-/blob/auto_prescription/data/io/dcm_utils.py
#################################################################

def translate_source_to_target_space(source, target, x, y, sl):
    '''
    Input source and target volume class.
    TODO - It is not clear what to enter for x, y, and sl.
    '''
    def translate_vertex(v, source_vol, target_vol):
        '''
        Converts vertex from in-space coordinates to out-space coordinates
        '''
        sample_lps = source_vol.xyz2lps(np.array(v)) # ~9-16 unit warhol
        sample_xyz = target_vol.lps2xyz(sample_lps)
        # xyz = target_vol.lps2xyz(lps)
        xyz = [sample_xyz[0], sample_xyz[1], int(round(sample_xyz[2]))]
        return xyz

    if isinstance(x, list) or isinstance(x, np.ndarray):
        out = []
        for i in range(len(x)):
            o = translate_vertex(v=[x[i], y[i], sl], source_vol=source, target_vol=target)
            out.append(o)
        return out
    elif isinstance(x, (float, int, np.float32, np.float64, np.uint8, np.uint16)):
        out = translate_vertex(v=[x, y, sl], source_vol=source, target_vol=target)
        #mt - should the code out.append(o) ???
        return out
    else:
        raise Exception('Unhandled point type: ' + str(type(x)))

# def execute():
#     tmp = np.load('./tmp.npz')
#     x = tmp['x']
#     y = tmp['y']
#     sl = int(tmp['sl'])
#     s_dir = '/data/mochila1/PatientStudy/UCSF/BACPAC0101001/02-20211110/lumbar_spine/8-LSSAG_T2/'
#     t_dir = '/data/mochila1/PatientStudy/UCSF/BACPAC0101001/02-20211110/lumbar_spine/10-LSAX_T2/'
#     source_to_target(s_dir, t_dir, x, y, sl)
#     return
    

class Volume:
    def __init__(self, dir, temp_dir=''): #orientation=None,
        ds_list = read_dicoms_from_directory(dir, False, temp_dir)

        # filter images that approximately match orientation
        ds_list1 = ds_list # keep all

        # get position
        ds1 = ds_list1[0]
        self.series_description = ds1.SeriesDescription.lower()
        self.position = np.array(ds1.ImagePositionPatient)
        self.rows = ds1.Rows
        self.columns = ds1.Columns
        self.slice_thickness = ds1.SliceThickness
        # self.accession_number = ds1.AccessionNumber
        # self.series_number = ds1.SeriesNumber

        # calculate dcos
        x_vec = np.array(ds1.ImageOrientationPatient[0:3])
        y_vec = np.array(ds1.ImageOrientationPatient[3:6])
        # TODO: Check on any issues with ImageOrientationPatient. Instead...
        # Use position_first - position_last to get z orientation (print out and check)

        num_images = len(ds_list1)
        self.slices = num_images
        
        pixel_spacing = np.array(ds1.PixelSpacing)
        position_first = np.array(ds1.ImagePositionPatient)
        
        ds_last = ds_list1[-1]
        position_last = np.array(ds_last.ImagePositionPatient)
        
        center_first = position_first \
                       + 0.5 * (ds1.Columns - 1) * pixel_spacing[0] * x_vec \
                       + 0.5 * (ds1.Rows - 1) * pixel_spacing[1] * y_vec

        center_last = position_last \
                      + 0.5 * (ds1.Columns - 1) * pixel_spacing[0] * x_vec \
                      + 0.5 * (ds1.Rows - 1) * pixel_spacing[1] * y_vec

        #print('num_images', num_images)
        #print('pixel_spacing:', pixel_spacing)
        #print('position_first:', position_first)
        #print('position_last:', position_last)
        #print('center_first:', center_first)
        #print('center_last:', center_last)
        # center = 0.5 * (center_first + center_last)
        #print('center:', center)

        z_vec = center_last - center_first
        z_len = np.linalg.norm(z_vec)
        z_vec = z_vec/z_len

        self.dcos = np.array([x_vec,y_vec,z_vec])

        # calculate pixel spacing
        if num_images == 1:
            pixel_spacing_z = ds1.SliceThickness
        else:
            pixel_spacing_z = z_len / (num_images - 1)

        self.pixel_spacing = [pixel_spacing[0], pixel_spacing[1], pixel_spacing_z]

        # load pixel data
        self.pixels = np.array([ds.pixel_array for ds in ds_list1]) # TODO: NO INT CAST

        # print('position:', self.position)
        # print('dcos:', self.dcos)
        # print('pixel spacing:', self.pixel_spacing)
        # print('pixels.shape:', self.pixels.shape)
        # print('rows:', self.rows)
        # print('columns:', self.columns)
        # print('slices:', self.slices)
        # print('accession_number:', self.accession_number)
        # print('series_number:', self.series_number)

    def lps2xyz(self, lps):
        xyz = np.array([0,0,0])
        lps1 = lps - self.position
        dcos = self.dcos
        for i in range(0,3):
            for j in range(0,3):
                xyz[i] += dcos[i][j]*lps1[j]
        #convert from mm to pixels
        xyz_px = xyz / self.pixel_spacing
        return xyz_px

    def xyz2lps(self, xyz):
        xyz_mm = xyz * self.pixel_spacing
        lps = np.array([0,0,0])
        dcos = self.dcos
        for i in range(0,3):
            for j in range(0,3):
                lps[i] += (dcos[j][i]*xyz_mm[j])
        lps1 = lps + self.position
        return lps1
        
class Volume_Slim:
    def __init__(self, dir, temp_dir=''): 
        ds_list = dcm_read_dir(dir, False, temp_dir)
        
        # try JPEG decompression if no results
        if len(ds_list)<1:
            ds_list = dcm_read_dir(dir, True, temp_dir)

        # collect slim metadata
        ds1 = ds_list[0]
        num_images = len(ds_list)
        self.slices = num_images
        self.rows = ds1.Rows
        self.columns = ds1.Columns
        try:
            self.series_description = ds1.SeriesDescription.lower()
        except:
            self.series_description = ''

        # load pixel data
        self.pixels = np.array([ds.pixel_array for ds in ds_list]) # TODO: NO INT CAST


