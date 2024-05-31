# Analysis scripts gaze-contingent VR phosphene simulation experiments.
This repository contains the analysis scripts for the publication:
De Ruyter Van Steveninck, J., Nipshagen, M., Van Gerven, M., Güçlü, U., Güçlüturk, Y., & Van Wezel, R. (2023). Gaze-contingent processing improves mobility performance and visual orientation in simulated head-steered prosthetic vision (Preprint). bioRxiv. https://doi.org/10.1101/2023.09.18.558225

Note this repository contains the data analysis. The code for the VR-simulations can be found in a separate repository: https://github.com/neuralcodinglab/spvgaze

## Usage 
The following steps can be performed for reproducing the analysis:
1. Dowload the experimental data from the Donders Institue data repository: https://doi.org/10.34973/gx4r-n774
2. Clone this repository.
4. Adjust the data directory in data_processing.py to match with your own storage location:
   
```python
DATA_DIR = r'D:\SPVGazeData'  # <- Replace this with your data directory
DATA_DIR_EXP1 = os.path.join(DATA_DIR, 'Experiment_1')
DATA_DIR_EXP2 = os.path.join(DATA_DIR, 'Experiment_2')
```

5. Run each of the notebook files to reproduce the different analyses.

### General notes
The analysis is based on the data of two separate VR experiments. Four types of data are used in the analysis:
1. Timeseries data from the eye-tracker (as provided by the manufacturers software): e.g., SingleEyeDataRecordC, containing the combined gaze data for both eyes.
2. Timeseries data from the Unity game engine: EngineDataRecord, containing the head positions, raycast vectors (pointing directions, trigger events (button presses), etc.
3. Trial configuration records saved after completion of each trial: TrialConfigRecord, containing the experimental conditions, the subject responses and the trial durations of each trial.
4. Additional data surveys (exit surveys) with personal responses.


Please refer to the paper and the notebook files for a full explanation on the data analyis.
