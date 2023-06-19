""" 
Module for loading, processing and saving the data files. Here, the processing involves
just basic unit conversion, and adding convenient labels to the dataframe columns.
"""

# Data management
import pandas as pd 
import os 

# Data processing and unit conversions
import numpy as np
import ast
import scipy
from scipy.spatial.transform import Rotation


## Defaults
BASE_PATH = os.path.dirname(__file__)
DATA_DIR = f'{BASE_PATH}/../_Datasets/SPVGazeData'          # The data directory
SUBJECTS = {'S35', 'S37', 'S38', 'S39', 'S40', # which subjects to include in the analysis
            'S41', 'S42', 'S45', 'S46', 'S47',
            'S49', 'S50', 'S51', 'S52', 'S53',
            'S54', 'S55', 'S56', 'S57'}        # Pilot subjects:"'S30', 'S30_','S31' 's33', 's33_', 'S34', 'S34_'"
LOAD_PREPROCESSED = True
DATA_KEYS = ['TrialConfigRecord', 'EngineDataRecord', 'SingleEyeDataRecordC'] # which kind of data to load
DOWNSAMPLE = 1 # for faster anaylis, the data can be downsampled with a factor



def expand_coordinates(df):
    """Reads coordinates (stored as tuples in the CSV) into separate columns"""
    for col in df.columns:
        if str(df[col][0]).startswith('('): # check if tuple
            if type(df[col][0]) == str: # check if tuple is stored as string
                df[col] = df[col].apply(ast.literal_eval) # convert string -> tuple
            n_dims = len(df[col][0]) # inspect first element: 2D, 3D or 4D coordinates
            suffix = 'WXYZ' if n_dims==4 else 'XYZ' # if 4D coordinates (quaternion), the first is W
            new_col_labels = [col+suffix[i] for i in range(n_dims)] # labels with suffix
            new_columns = pd.DataFrame(df[col].to_list(), index=df.index) # expand tuples to separate colunms
            df[new_col_labels] = new_columns

def add_conditions_from_trial_data(df, trial_data, conditions):
    """Lookup the trial conditions in trial_data and add to df"""
    for condition in conditions:
        lookup = trial_data.set_index('TrialIdentifier')[condition].to_dict() # map ID -> condition
        df[condition] = df.TrialIdentifier.replace(lookup) # add condition to df

def quat_to_euler(x):
    if type(x) == str:
        quat_tuple = ast.literal_eval(x) # convert string -> tuple
    quat_obj = Rotation.from_quat(quat_tuple)
    # return quat_obj.as_euler('xyz', degrees=True)
    return quat_obj.as_euler('XYZ', degrees=True)

def quat_to_dir(x):
    if type(x) == str:
        quat_tuple = ast.literal_eval(x) # convert string -> tuple
    quat_obj = Rotation.from_quat(quat_tuple)
    return tuple(quat_obj.apply([0,0,1]))

def preprocess_data(exp_data, trial_configs=None):
    if trial_configs is None:
        trial_configs = exp_data['TrialConfigRecord']

    for data_key in exp_data.keys():
        dataframe = exp_data[data_key]

        # Copy some useful columns from the TrialConfigRecord dataframe to the other dataframes
        if data_key != 'TrialConfigRecord':
            add_conditions_from_trial_data(dataframe,
                                         trial_configs,
                                         ['ExperimentalTask', 'Block', 'GazeCondition', 'Subject'])

        # Convert Quaternions to normalized direction vector (in separate column)
        rot_cols = [col for col in dataframe.columns if 'Rot' in col]
        if rot_cols:
            dir_cols = [col.replace('Rot', 'Dir') for col in rot_cols]
            dataframe[dir_cols] = dataframe.apply({col: quat_to_dir for col in rot_cols})

        # Put coordinates in separate columns (instead of tuples)
        expand_coordinates(dataframe)

    return

def save_preprocessed_data(exp_data, calibr_data, data_dir):
    save_dir = os.path.join(data_dir,'_preprocessed')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for data_key, dataframe in exp_data.items():
        fn = os.path.join(save_dir,f'{data_key}.tsv')
        dataframe.to_csv(fn, sep = '\t', index=False)

    for data_key, dataframe in calibr_data.items():
        fn = os.path.join(save_dir,f'calibrationTest_{data_key}.tsv')
        dataframe.set_index('TrialIdentifier').to_csv(fn, sep = '\t', index=False)

def load_preprocessed_data(path):
    exp_data, calibr_data = dict(), dict()
    files = os.listdir(path)
    for fn in files:
        data_key = fn.replace('calibrationTest_', '').replace('.tsv', '')
        if 'calibration' in fn:
            calibr_data[data_key] = pd.read_csv(os.path.join(path,fn),sep='\t')
        else:
            exp_data[data_key] = pd.read_csv(os.path.join(path,fn),sep='\t')
    return exp_data, calibr_data

def get_filenames(subjects, data_dir, data_keys=None):
    """Returns dict with filenames for all specified subjects found in the data 
    directory. The filenames are indexed by record type. """
    # Initialize dict with filenames (categorized by record type)
    if data_keys is None:
        data_keys = ['TrialConfigRecord', 'SingleEyeDataRecordR', 'SingleEyeDataRecordL',
                    'SingleEyeDataRecordC', 'EyeTrackerDataRecord', 'EngineDataRecord']
    data_files = {k: [] for k in data_keys}
    calibr_files = {k: [] for k in data_keys}

    # Add all files per subject
    for subj in subjects:
        base_path = os.path.join(data_dir,subj)
        listdir = sorted([os.path.join(base_path,fn) for fn in os.listdir(base_path) if 'tsv' in fn])
        for key in data_keys:
            data_files[key] += [fn for fn in listdir if (key in fn) and (not 'calibration' in fn)]
            calibr_files[key] += [fn for fn in listdir if (key in fn) and ('calibration' in fn)]
    calibr_files = {k: v for k, v in calibr_files.items() if v} # remove empty keys
    return data_files, calibr_files

def load_data_from_filenames(filenames_dict, downsample=None, **pd_kwargs):
    """For each item in filenames dict, reads the listed TSV files and concatenate
     in a single dataframe. Returns dict with pandas dataframes."""
    # The output dict
    exp_data = dict()
    downsample_rate = 1 if downsample is None else downsample

    # Loop through data keys ('TrialConfigRecord', 'SingleEyeDataRecordC', 'EngineDataRecord', etc.)
    for data_key, filenames in filenames_dict.items():

        # Put all trials (for all subjects) in a single dataframe
        dataframe = None
        for i in range(len(filenames)):

            # Read new rows from file
            new = pd.read_csv(filenames[i],sep = '\t', **pd_kwargs)
            
            if data_key in ['EngineDataRecord', 'SingleEyeDataRecordR', 'SingleEyeDataRecordL', 'SingleEyeDataRecordC']:
                new = new[::downsample_rate].copy()

            # Get block number, trial and subject from filename and directory
            path, fn = os.path.split(filenames[i])
            subject = os.path.basename(path).replace('_','')
            block = int(fn[0:2])
            relative_trial_number = int(fn[3:5])
            identifier = f'{subject}B{block}T{relative_trial_number}'
            new['Subject'] = subject
            new['Block'] = block
            new['RelativeTrialNumber'] = relative_trial_number
            new['TrialIdentifier'] = identifier

            # Append new rows to the dataframe
            if dataframe is None:
                dataframe = new
            else:
                dataframe = pd.concat([dataframe,new], ignore_index = True)

        # Add the dataframe to the output dict
        exp_data[data_key] = dataframe
    return exp_data


def get_target_locations(fn = os.path.join(DATA_DIR,'_TargetLocations.tsv'), sep='\t'):
    return pd.read_csv(fn, sep=sep).set_index('TargetName')

def get_survey_responses(fn = os.path.join(DATA_DIR,'_ExitSurvey.tsv'), sep='\t', subjects=list(SUBJECTS)):
    data = pd.read_csv(fn, sep=sep, encoding='utf-16')
    question_mapping = data.head(1).T.to_dict()[0]
    responses = data[2:].set_index('Q1').loc[subjects].copy()
    responses.index.name = 'Subject'
    return responses, question_mapping