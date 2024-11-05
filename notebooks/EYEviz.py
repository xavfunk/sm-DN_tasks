import pandas as pd
import matplotlib.pyplot as pl
from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from scipy.stats import sem
import os
import yaml
from pathlib import Path
from collections import defaultdict as dd
from linescanning import (
    dataset,
    plotting,
    fitting,
    utils
)
from math import atan2, degrees
# Import the packages you need
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import shutil
import seaborn
import statistics
import glob
import dill as pickle


opj = os.path.join

def rp(str):
    sstr = str.replace('_','-')
    return sstr


def getExpAttr(clazz):
    return [name for name, attr in clazz.__dict__.items()
            if not name.startswith("__") 
            and not callable(attr)
            and not type(attr) is staticmethod
            and not type(attr) is str]


class EYEviz():

    def __init__(self, log_paths, out_path,sub_med):

        pl.rcParams.update({'font.size': 16})
        pl.rcParams.update({'pdf.fonttype':42})
        pl.rcParams.update({'figure.max_open_warning': 0})
        pl.rcParams['axes.spines.right'] = False
        pl.rcParams['axes.spines.top'] = False

        self.out_path = out_path # The path were the output of the function is saved

        for log_path in [l_p for l_p in log_paths if os.path.isdir(l_p)]: # Loop over different paths that were given as input

            id_str = [el.replace('-','_') for el in log_path.split(os.sep)[-1].split('_')]

            print(log_path)

            try:
                settings_file = [str(path) for path in Path(log_path).glob('*.yml')][0] # Find settings file
                
             
                events_file = [str(path) for path in Path(log_path).glob('*.tsv')][0] # Find events file
            

                eyetrack_file = [str(path) for path in Path(log_path).glob('*.edf')][0] # Find eyetracking data file

            except:
                print('no eyetrack or event or settings file found')
                continue

            with open(settings_file) as f:
                settings_dict = yaml.safe_load(f) # Load settingsfile
  

            df = pd.read_csv(events_file, sep="\t") # Load eventsfile
            df_responses = df[df['event_type'] == 'Response']

            
                
            try: # Here the function checks if there is already a pickled file in the folder, if there is not, it loads in the eyetracking data and saves it as pickle
                pkl_file = [str(path) for path in Path(log_path).glob('*.pkl')][0]
                eyetrack_data = pickle.load(open(pkl_file, 'rb'))
            except:
                print('No pickled eyetracking file')
                try:
                    eyetrack_data = dataset.ParseEyetrackerFile(eyetrack_file, use_bids = True, verbose = True)
                    pickle.dump(eyetrack_data, open(log_path + f'/df_eyetrack_{id_str[0]}_{id_str[1]}_{id_str[2]}_{id_str[3]}.pkl', 'wb'))
                except: # If there is no eyetracking data at all in the folder, it skips this run
                    eyetrack_data = 0
                    print('No eyetracking data found')

         
            if eyetrack_data == 0:
                print('No eyetracking data found')
                continue  # Skip run

            df_gaze = eyetrack_data.df_space_eye
            df_gaze = pd.DataFrame(df_gaze)
            
            if df_gaze.empty:
                print('No eyetracking data found')
                continue

            
            df_gaze = df_gaze.reset_index()
            
            # Select left and right eye
            df_gaze_left = utils.select_from_df(df_gaze, expression = 'eye = L')
            df_gaze_right = utils.select_from_df(df_gaze, expression = 'eye = R')
            # Time
            if np.any(df_gaze_left):
                t = df_gaze_left.t.values  #Timings to plot
            elif np.any(df_gaze_right):
                t = df_gaze_right.t.values

            # Calculating visual angle

            dis_to_screen = settings_dict['monitor']['distance'] # Distance from participant to the screen in cm
            size_scr_hor = 1920 # width of the screen in pixels
            size_scr_ver = 1080 # height of the screen in pixels
            monitor_width = 69.8 # width of the screen in cm
            x_cor_fixcross = size_scr_hor/2 # x-coordinate of the fixation cross
            y_cor_fixcross = size_scr_ver/2 # y-coordinate of the fixation cross

            x_coor_pix_left = df_gaze_left.loc[:,"gaze_x_int"] # Eyetracking data for gaze
            y_coor_pix_left = df_gaze_left.loc[:,"gaze_y_int"]
            x_coor_pix_right = df_gaze_right.loc[:,"gaze_x_int"] # Eyetracking data for gaze
            y_coor_pix_right = df_gaze_right.loc[:,"gaze_y_int"]
            x_coor_relfix_left = (x_coor_pix_left - x_cor_fixcross) # Pixels relative to the fixation cross
            y_coor_relfix_left = (y_coor_pix_left - y_cor_fixcross)
            x_coor_relfix_right = (x_coor_pix_right - x_cor_fixcross) # Pixels relative to the fixation cross
            y_coor_relfix_right = (y_coor_pix_right - y_cor_fixcross)

            deg_per_pix = degrees(atan2(.5*monitor_width,dis_to_screen))/(.5*size_scr_hor) # How many degrees is one pixel
      
            visual_angle_x_left = np.array(x_coor_relfix_left*deg_per_pix) # x coordinate visual angle for left eye
            visual_angle_y_left = np.array(y_coor_relfix_left*deg_per_pix) # y coordinate visual angle for left eye
            eccentricity_left = (visual_angle_x_left**2 + visual_angle_y_left**2) **0.5 # Eccentricity for left eye
            visual_angle_x_right = np.array(x_coor_relfix_right*deg_per_pix) # x coordinate visual angle for right eye
            visual_angle_y_right = np.array(y_coor_relfix_right*deg_per_pix) # y coordinate visual angle for left eye
            eccentricity_right = (visual_angle_x_right**2 + visual_angle_y_right**2) **0.5 # Eccentricity for right eye

            # Pupil size
            pupil_size_left = df_gaze_left["pupil_int"].values
            pupil_size_right = df_gaze_right["pupil_int"].values

            # If data is binocular, calculate the mean of the left and right eye
            if (np.any(df_gaze_left)) & (np.any(df_gaze_right)):
                visual_angle_x_bi = np.mean([visual_angle_x_left, visual_angle_x_right],axis = 0) # x coordinate - mean of left and right eye
                visual_angle_y_bi = np.mean([visual_angle_y_left, visual_angle_y_right],axis = 0) # y coordinate - mean of left and right eye
                eccentricity_bi = (visual_angle_x_bi**2 + visual_angle_y_bi**2) **0.5 # Eccentricity - mean of left and right eye

            if not hasattr(self, id_str[2]):
                setattr(self, id_str[2], EYEviz.Task())

            this_task = getattr(self, id_str[2])

            if not hasattr(this_task, id_str[0]):
                setattr(this_task, id_str[0], EYEviz.Subject())

            this_subject = getattr(this_task, id_str[0])          

            if not hasattr(this_subject, id_str[1]):
                setattr(this_subject, id_str[1], EYEviz.Session())

            this_session = getattr(this_subject, id_str[1]) 

            if not hasattr(this_session, id_str[3]):
                setattr(this_session, id_str[3], EYEviz.Run())
            
            this_run = getattr(this_session, id_str[3])

            # Attribute values that are needed later in the function
            this_run.expsettings = settings_dict
            this_run.lefteye_x = visual_angle_x_left
            this_run.lefteye_y = visual_angle_y_left
            this_run.lefteye_ecc = eccentricity_left
            this_run.righteye_x = visual_angle_x_right
            this_run.righteye_y = visual_angle_y_right
            this_run.righteye_ecc = eccentricity_right
            if (np.any(df_gaze_left)) & (np.any(df_gaze_right)):
                this_run.bieye_x = visual_angle_x_bi  
                this_run.bieye_y = visual_angle_y_bi
                this_run.bieye_ecc = eccentricity_bi
            this_run.time = t
            this_run.df_responses = df_responses
            this_run.log_path = log_path
            this_run.eventsfile = df
            this_run.deg_per_pix = deg_per_pix
            this_run.x_cor_fixcross = x_cor_fixcross
            this_run.y_cor_fixcross = y_cor_fixcross
            this_run.sub_med = sub_med
            this_run.pupil_left = pupil_size_left
            this_run.pupil_right = pupil_size_right

            this_run.saccades = eyetrack_data.df_saccades
            this_run.blinks = eyetrack_data.df_blinks
            this_run.sample_rate = eyetrack_data.sample_rate

            os.makedirs(opj(self.out_path,rp(id_str[0]),rp(id_str[1])),exist_ok=True)

            this_run.out_path = opj(self.out_path,rp(id_str[0]),rp(id_str[1]))

            if 'sub' not in this_run.expsettings or this_run.expsettings['sub'] != id_str[0]:
                this_run.expsettings['sub'] = id_str[0]
                #print(f'warning: subject in filename and expsettings {settings_file} are different. using filename')
                
            if 'ses' not in this_run.expsettings or this_run.expsettings['ses'] != id_str[1]:
                this_run.expsettings['ses'] = id_str[1]
                #print(f'warning: session in filename and expsettings {settings_file} are different. using filename')

            if 'task' not in this_run.expsettings or this_run.expsettings['task'] != id_str[2]:
                this_run.expsettings['task'] = id_str[2]
                #print(f'warning: task in filename and expsettings {settings_file} are different. using filename')

            if 'run' not in this_run.expsettings or this_run.expsettings['run'] != id_str[3]:
                this_run.expsettings['run'] = id_str[3]
                #print(f'warning: task in filename and expsettings {settings_file} are different. using filename') 

    def subset_all(self):
        # This function makes the subsets for all the different runs
        print("Creating subsets...")
        for task in getExpAttr(self):
            
            for subject in getExpAttr(getattr(self,task)):
                for session in getExpAttr(getattr(getattr(self,task),subject)):
                    for run in getExpAttr(getattr(getattr(getattr(self,task),subject),session)):

                        this_run = getattr(getattr(getattr(getattr(self,task),subject),session),run)


                        ind_TR = this_run.eventsfile['response'] == 't'
                        onsets_TR = np.array(this_run.eventsfile['onset'][ind_TR]) 

                        if 'CD' in task: # Low spatial frequency and high spatial frequency
                            
                            this_run.CDsubset()
                        elif 'CS' in task: # Surrounding vs. no surrounding
                            
                            this_run.CSsubset()
                        elif 'EH' in task: # Surrounding vs. no surrounding

                            this_run.EHsubset()   

                        elif ('2R' in task) & (len(onsets_TR) >= 254): # Barpass vs. interval between the barpasses
                            this_run.pRFsubset()


    def plot_all(self):
        # This function makes plots for all runs
        print("plotting data...")
        for task in getExpAttr(self):
            this_task = getattr(self,task)

            for subject in getExpAttr(this_task):
                this_subject = getattr(this_task,subject)
                for session in getExpAttr(this_subject):
                    this_session = getattr(this_subject,session)
                    for run in getExpAttr(this_session):
                        this_run = getattr(this_session, run)

                        ind_TR = this_run.eventsfile['response'] == 't'
                        onsets_TR = np.array(this_run.eventsfile['onset'][ind_TR]) 
            

                        this_run.PupilPlot() # Plot the pupil size over the run, same for each task
                        if 'CD' in task:
                            this_run.CDplot() # Plots gaze for contrast discrimination task
                            
                        elif 'CS' in task:
                            this_run.CSplot() # Plots gaze for center surround task
                        elif 'EH' in task:
                            this_run.EHplot() # Plots gaze for ebbinghaus task
                        elif ('2R' in task) & (len(onsets_TR) >= 254):
                            this_run.pRFplot()  # Plots gaze for pRF task

    def plot_group(self): # does not plot yet, but should make plots eventually
        for task in getExpAttr(self):
            this_task = getattr(self,task)

            this_task.SavePupilGroup() # Saves the pupil size data in a pickle file

            this_task.SaveEyeGroup() # Saves the gaze data in a pickle file

    class Task():
        def __init__(self):
            pass

        def SavePupilGroup(self):

            PupilSub = dd(list)
            TimeDict = dd(list)

            for su, subject in enumerate(getExpAttr(self)):
                this_subject = getattr(self,subject)
                

                for se,session in enumerate(getExpAttr(this_subject)):
                    this_session = getattr(this_subject,session)
                    for rr,run in enumerate(getExpAttr(this_session)):
                        this_run = getattr(this_session, run)

                    

                        if np.any(this_run.pupil_left):
                            Pupil = this_run.pupil_left
                            eye = 'left'
                        elif np.any(this_run.pupil_right):
                            Pupil = this_run.pupil_right
                            eye = 'right'
                        elif (np.any(this_run.pupil_left)) & (np.any(this_run.pupil_right)):
                            Pupil = np.mean([this_run.pupil_left, this_run.pupil_right],axis = 0)
                            eye = 'both'
                        # Save data for pupil and time in a dictionairy
                        PupilSub[subject].append(Pupil)
                        TimeDict[subject].append(this_run.time)

                        # NOT FINISHED. Goal was to make a function that plots the average pupil size for each task. Another function should plot the average standard deviation for the pupil size for each task
                        # Dose is now hardcoded in the filename, should be changed to go automatically. 
            pickle.dump(PupilSub, open('/data1/projects/dumoulinlab/Lab_members/Marco/SM-pRF/derivatives/eyetrack_pupilsize/' + f"pupilsize_participants_10mg_{this_run.expsettings['task']}.pkl", 'wb'))
            pickle.dump(TimeDict, open('/data1/projects/dumoulinlab/Lab_members/Marco/SM-pRF/derivatives/eyetrack_pupilsize/' + f"time_participants_10mg_{this_run.expsettings['task']}.pkl", 'wb'))
            return

        
        def SaveEyeGroup(self):

            GazeEyeDict = dd(lambda:dd(list))
            # TimeDictEye = dd(list)

            for su, subject in enumerate(getExpAttr(self)):
                this_subject = getattr(self,subject)
                

                for se,session in enumerate(getExpAttr(this_subject)):
                    this_session = getattr(this_subject,session)
                    for rr,run in enumerate(getExpAttr(this_session)):
                        this_run = getattr(this_session, run)

                        if np.any(this_run.lefteye_x):
                            gaze_x = this_run.lefteye_x
                            gaze_y = this_run.lefteye_y
                            gaze_ecc = this_run.lefteye_ecc
                            eye = 'left'
                        elif np.any(this_run.pupil_right):
                            gaze_x = this_run.righteye_x
                            gaze_y = this_run.righteye_y
                            gaze_ecc = this_run.righteye_ecc
                            eye = 'right'
                        elif (np.any(this_run.pupil_left)) & (np.any(this_run.pupil_right)):
                            gaze_x = this_run.bieye_x
                            gaze_y = this_run.bieye_y
                            gaze_ecc = this_run.bieye_ecc
                            eye = 'both'

                        # Saves the data for x position, y position and eccentricity of gaze for each subject
                        GazeEyeDict['x'][subject].append(gaze_x)
                        GazeEyeDict['y'][subject].append(gaze_y)
                        GazeEyeDict['ecc'][subject].append(gaze_ecc)
                        # TimeDictEye[subject].append(this_run.time)

                        # NOT FINISHED. Goal was to plot the average standard deviation of the gaze for each task. Another function should plot the mean gaze for each task. 
            # Dose is now hardcoded in filename, should be changed to go automatically
            pickle.dump(GazeEyeDict, open('/data1/projects/dumoulinlab/Lab_members/Marco/SM-pRF/derivatives/eyetrack_pupilsize/' + f"gaze_participants_10mg_{this_run.expsettings['task']}.pkl", 'wb'))
            # pickle.dump(TimeDictEye, open('/data1/projects/dumoulinlab/Lab_members/Marco/SM-pRF/derivatives/psychophysics/groups/placebo guess/' + f"time_participants_placebo_{this_run.expsettings['task']}.pkl", 'wb'))
            return
                        
    class Subject():
        def __init__(self):
            pass
    class Session():
        def __init__(self):
            pass
    class Run():
        def __init__(self):
            pass
        
        def CDsubset(self):

            # Get the onsets and phases from the TSV file
            onsets = self.eventsfile['onset'] # Onsets of the events
            phases = self.eventsfile['phase'] # Which phase
            event = self.eventsfile['event_type'] # Is a stimulus shown?
            frames = self.eventsfile['nr_frames'] # Frames shown in experiment

            #all trials
            ind = np.where(((phases == 0) & (event == 'stim') & (frames > 0))) # Stimulus onset
            ind = ind[0]
            if frames.iloc[-1] > 0:
                ind = ind[:-1]
            ind_offset = ind + 3 # Stimulus offset
            onsets_stim_s =  np.array(onsets[ind])
            offsets_stim_s =  np.array(onsets[ind_offset])

            # read in relevant column to subset trials with different spatial frequencies
            spat_freq = self.eventsfile['spatial_frequency_cycles']
            spat_freq_uni = spat_freq.unique()

            # high spatial frequency
            high_sf = np.where((spat_freq == spat_freq_uni[0]) & (phases == 0) & (event == 'stim') & (frames > 0))
            high_sf = high_sf[0]
            if frames.iloc[-1] > 0:
                high_sf = high_sf[:-1]
            high_sf_off = high_sf + 3
            time_high_sf = np.array(onsets[high_sf])
            time_high_sf_off = np.array(onsets[high_sf_off])

            # low spatial frequency
            low_sf = np.where((spat_freq == spat_freq_uni[2]) & (phases == 0) & (event == 'stim') & (frames > 0))
            low_sf = low_sf[0]
            if frames.iloc[-1] > 0:
                low_sf = low_sf[:-1]
            low_sf_off = low_sf + 3
            time_low_sf = np.array(onsets[low_sf])
            time_low_sf_off = np.array(onsets[low_sf_off])
            # Define onset and offset times for each subset, this is used later to create a mask to select the data for each subset
            time_subsets_on = dict() # Onset times
            time_subsets_off = dict() # Offset times

            # Subsets
            all_subset_names = ['low spatial freq','high spatial freq'] # All the subsets from the data
            current_subset_names = ['low spatial freq', 'high spatial freq'] # The subsets you want to plot

            # Positions
            positions = ['x_pos_L','y_pos_L','ecc_L','x_pos_R','y_pos_R','ecc_R','x_pos_bi','y_pos_bi','ecc_bi']
            mean_pos = ['mean_x_L','mean_y_L','mean_ecc_L','mean_x_R','mean_y_R','mean_ecc_R','mean_x_bi','mean_y_bi','mean_ecc_bi']
            median_pos = ['median_x_L','median_y_L','median_ecc_L','median_x_R','median_y_R','median_ecc_R','median_x_bi','median_y_bi','median_ecc_bi']
            var_pos = ['var_x_L','var_y_L','var_ecc_L','var_x_R','var_y_R','var_ecc_R','var_x_bi','var_y_bi','var_ecc_bi']

            # From which eye would you want to plot the data? left, right or binocular
            if np.any(self.lefteye_x):
                self.eye = "L"
            elif np.any(self.righteye_x):
                self.eye = "R"
            elif (np.any(self.lefteye_x)) and (np.any(self.righteye_x)):
                self.eye = "bi"

            # Find the onset and offset times for the subsets

            for subset_name in all_subset_names:

                if subset_name == 'low spatial freq':
                    time_subsets_on[subset_name] = time_low_sf # Onset times
                    time_subsets_off[subset_name] = time_low_sf_off # Offset times

                elif subset_name == 'high spatial freq':
                    time_subsets_on[subset_name] = time_high_sf # Onset times   
                    time_subsets_off[subset_name] = time_high_sf_off # Offset times

        

            subsets = dd(lambda:dd(list))

            for subset_name in current_subset_names:

                time_subset_on = time_subsets_on[subset_name] # Onset times
                time_subset_off = time_subsets_off[subset_name] # Offset times


                for i in range(len(time_subset_on)):

                    mask_trial = (self.time>=time_subset_on[i]) & (self.time<time_subset_off[i]) # mask to select data for each subset

                    # Select data for each subset, check if there is data for the left or the right eye
                    if np.any(self.lefteye_x):
                        subsets[subset_name]['x_pos_L'].append(self.lefteye_x[mask_trial])
                        subsets[subset_name]['y_pos_L'].append(self.lefteye_y[mask_trial])
                        subsets[subset_name]['ecc_L'].append(self.lefteye_ecc[mask_trial])
                        self.eye = "L"
                    if np.any(self.righteye_x):
                        subsets[subset_name]['x_pos_R'].append(self.righteye_x[mask_trial])
                        subsets[subset_name]['y_pos_R'].append(self.righteye_y[mask_trial])
                        subsets[subset_name]['ecc_R'].append(self.righteye_ecc[mask_trial])
                        self.eye = "R"
                    if (np.any(self.lefteye_x)) and (np.any(self.righteye_x)):
                        subsets[subset_name]['x_pos_bi'].append(self.bieye_x[mask_trial])
                        subsets[subset_name]['y_pos_bi'].append(self.bieye_y[mask_trial])
                        subsets[subset_name]['ecc_bi'].append(self.bieye_ecc[mask_trial])
                        self.eye = "bi"

                    subsets[subset_name]['duration'].append(np.linspace(0,0.001*len(mask_trial),len(mask_trial)))


                # Make everything the same length for calculating the mean and the median
                min_len = np.min([len(el) for el in subsets[subset_name]['x_pos_R']])

                for position,mean_position,median_position,var_position in zip(positions,mean_pos,median_pos,var_pos):

                    # Make all the subsets the same length
                    subsets[subset_name][position] = np.array([el[:min_len] for el in subsets[subset_name][position]])
                    subsets[subset_name][var_position] = np.array([np.var(trial) for trial in subsets[subset_name][position]])

                    
                    # Calculate the median
                    subsets[subset_name][median_position] = np.median(subsets[subset_name][position],axis = 0)

                    # If the user of the function wants the median to be subtracted from the data, the function enters this statement. In that case, the median is subtracted
                    # The median is calculated over the trials. This is subtracted from each trial.
                    if self.sub_med == 'True':
        
                        for j in range(len(subsets[subset_name][position])):
                            subsets[subset_name][position][j] = [element1 - element2 for (element1,element2) in zip(subsets[subset_name][position][j,:], subsets[subset_name][median_position])]
                            # subsets[subset_name][position][j,:] == np.subtract(subsets[subset_name][position][j,:],subsets[subset_name][median_position]) 
                
                    # Calculate the mean
                    subsets[subset_name][mean_position] = np.mean(subsets[subset_name][position],axis = 0)
                
                subsets[subset_name]['duration'] = np.array([el[:min_len] for el in subsets[subset_name]['duration']])

            self.subsets = subsets 

        def CDplot(self):
            
            pl.rcParams.update({'font.size': 16})
            pl.rcParams.update({'pdf.fonttype':42})
            pl.rcParams.update({'figure.max_open_warning': 0})
            pl.rcParams['axes.spines.right'] = False
            pl.rcParams['axes.spines.top'] = False 

            # Create lists to loop over for plotting
            eye = self.eye # Select which eye is tracked
            pos_to_plot = [f"x_pos_{eye}",f"y_pos_{eye}",f"ecc_{eye}"]
            mean_to_plot = [f"mean_x_{eye}",f"mean_y_{eye}",f"mean_ecc_{eye}"]
            median_to_plot = [f"median_x_{eye}",f"median_y_{eye}",f"median_ecc_{eye}"]
            var_to_plot = [f"var_x_{eye}",f"var_y_{eye}",f"var_ecc_{eye}"]
            current_subset_names = ['low spatial freq', 'high spatial freq']

            fig,ax = pl.subplots(3,len(current_subset_names),constrained_layout = True,figsize = (15,15)) # Create figure
            position_names = ['x','y','eccentricity']
            # Plot - rows x, y and eccentricity and one column for every subset
            for i,subset in enumerate(current_subset_names): # Loop over subsets
                for k,position,var_position in zip(range(len(pos_to_plot)), pos_to_plot,var_to_plot): # Loop over positions to plot (x, y, eccentricity for selected eye)
                    for dur,j in zip(self.subsets[subset]['duration'],self.subsets[subset][position]): # Loop over trials
                        ax[k,i].plot(np.array(dur),np.array(j),linewidth = 0.2, color = 'black', alpha = 0.1)
            
         

                    mean_plot, = ax[k,i].plot(np.array(self.subsets[subset]['duration'][0,:]),np.array(self.subsets[subset][mean_to_plot[k]]),linewidth = 3, color = 'red', label = 'mean')
                    # median_plot, = ax[k,i].plot(np.array(self.subsets[subset]['duration'][0,:]),np.array(self.subsets[subset][median_to_plot[k]]),linewidth = 3, color = 'blue', label = 'median')

                    quantile_10 = np.quantile(self.subsets[subset][var_position],0.90)
                    quantile_25 = np.quantile(self.subsets[subset][var_position],0.75)

                    var_small = np.sum((self.subsets[subset][var_position] < 1))
                    var_med = np.sum((self.subsets[subset][var_position] > 1) & (self.subsets[subset][var_position] < 2))
                    var_large = np.sum((self.subsets[subset][var_position] > 2))
                    trials = len(self.subsets[subset][var_position])


                    ax[k,i].set_title(f'{subset} - {position_names[k]} position')
                    ax[k,i].set_xlabel('Time (s)')
                    ax[k,i].set_ylabel('Visual angle (degrees)')
                    ax[k,i].set_xlim([0,np.max(self.subsets[subset]['duration'])])
                    ax[k,i].text(0.1,2.75,f'median = {np.mean(np.array(self.subsets[subset][median_to_plot[k]])):.4f}') 

                    # Set limits of the y axis. If the median is subtracted, the eccentricity can be negative.
                    # If the median is not subtracted, the eccentricity cannot be negative.
                    # Therefore, the y-axis goes to negative values if the median is subtracted, but not if it is not subtracted.  
                    if k != 2:
                        ax[k,i].set_ylim([-3,3])
                    elif k == 2:
                        if self.sub_med == 'True':
                            ax[k,i].set_ylim([-3,3])
                        elif self.sub_med == 'False':
                            ax[k,i].set_ylim([0,3])

                # Plot the coordinates of the stimulus
                for coordinate in range(2): 
                    stim_cen = ax[coordinate,i].axhline(-2.5,linewidth = 1,linestyle = '--',label = 'stimulus center') 
                    stim_cen = ax[coordinate,i].axhline(2.5,linewidth = 1,linestyle = '--') 
                    stim_ext = ax[coordinate,i].axhline(-2.5 - 1.25,linewidth = 1,linestyle = '--',alpha = 0.8,label = 'stimulus extent') 
                    stim_ext = ax[coordinate,i].axhline(-2.5 + 1.25,linewidth = 1,linestyle = '--',alpha = 0.8)
                    stim_ext = ax[coordinate,i].axhline(2.5 - 1.25,linewidth = 1,linestyle = '--',alpha = 0.8) 
                    stim_ext = ax[coordinate,i].axhline(2.5 + 1.25,linewidth = 1,linestyle = '--',alpha = 0.8)
            
            pl.suptitle(f"Gaze task: CD sub: {self.expsettings['sub']} ses: {self.expsettings['ses']} run: {self.expsettings['run']} eye: {eye}",fontsize = 24)
            
            fig.legend([mean_plot],['mean'],loc = 'center right',bbox_to_anchor = (1.2,0.5)) 
            fig.savefig(self.out_path + f"/{self.expsettings['sub']}_{self.expsettings['ses']}_{self.expsettings['run']}_{self.expsettings['task']}_gaze.pdf", dpi=600, bbox_inches='tight', transparent=True)

            return
        
        def EHsubset(self):

            # Get the onsets and phases from the TSV file
            onsets = self.eventsfile['onset'] # Onsets of the events
            phases = self.eventsfile['phase'] # Which phase
            event = self.eventsfile['event_type'] # Is a stimulus shown?
            frames = self.eventsfile['nr_frames'] # Frames shown in experiment

            # all trials
            ind = np.where((phases == 0) & (event == 'stim') & (frames > 0)) # Stimulus onset
            ind = ind[0]
            if frames.iloc[-1] > 0:
                ind = ind[:-1]
            ind_offset = ind + 3 # Stimulus offset


            # read in relevant column to subset trials with different surrounds
            sur_rad = self.eventsfile['Surround stim radius']
            sur_rad_uni = np.unique(sur_rad)

            # no surrounds
            no_sur = np.where((sur_rad == sur_rad_uni[0]) & (phases == 0) & (event == 'stim') & (frames > 0)) # Onsets
            no_sur = no_sur[0]
            if frames.iloc[-1] > 0:
                no_sur = no_sur[:-1]
            no_sur_off = no_sur + 3 # Offsets
            time_no_sur =  np.array(onsets[no_sur])
            time_no_sur_off =  np.array(onsets[no_sur_off])

            # big surrounds
            big_sur = np.where((sur_rad == sur_rad_uni[1]) & (phases == 0) & (event == 'stim') & (frames > 0))
            big_sur = big_sur[0]
            if frames.iloc[-1] > 0:
                big_sur = big_sur[:-1]
            big_sur_off = big_sur + 3
            time_big_sur = np.array(onsets[big_sur])
            time_big_sur_off = np.array(onsets[big_sur_off])

            # Subsets
            all_subset_names = ['no surr','large surr'] # All the subsets from the data
            current_subset_names = ['no surr', 'large surr'] # The subsets you want to plot

            # Positions
            positions = ['x_pos_L','y_pos_L','ecc_L','x_pos_R','y_pos_R','ecc_R','x_pos_bi','y_pos_bi','ecc_bi']
            mean_pos = ['mean_x_L','mean_y_L','mean_ecc_L','mean_x_R','mean_y_R','mean_ecc_R','mean_x_bi','mean_y_bi','mean_ecc_bi']
            median_pos = ['median_x_L','median_y_L','median_ecc_L','median_x_R','median_y_R','median_ecc_R','median_x_bi','median_y_bi','median_ecc_bi']
            var_pos = ['var_x_L','var_y_L','var_ecc_L','var_x_R','var_y_R','var_ecc_R','var_x_bi','var_y_bi','var_ecc_bi']

            # Make a dictionairy for the onsets and offsets of the trials
            time_subsets_on = dict()
            time_subsets_off = dict()

            for subset_name in all_subset_names:

                if subset_name == 'no surr': # Onsets and offsets for trials without surrounding
                    time_subsets_on[subset_name] = time_no_sur
                    time_subsets_off[subset_name] = time_no_sur_off

                elif subset_name == 'large surr': # Onsets and offsets for trials with surrounding
                    time_subsets_on[subset_name] = time_big_sur
                    time_subsets_off[subset_name] = time_big_sur_off

            subsets = dd(lambda:dd(list)) # Dictionairy to save the data per subset 

            for subset_name in current_subset_names: # Loop over subsets

                time_subset_on = time_subsets_on[subset_name] # Onset times
                time_subset_off = time_subsets_off[subset_name] # Offset times


                for i in range(len(time_subset_on)): # Loop over trials

                    mask_trial = (self.time>=time_subset_on[i]) & (self.time<time_subset_off[i]) # mask to select data for each trial for this subset

                    # Select data for each subset, for the x position, y position and eccentricit and for both eyes. 
                    if np.any(self.lefteye_x):
                        subsets[subset_name]['x_pos_L'].append(self.lefteye_x[mask_trial])
                        subsets[subset_name]['y_pos_L'].append(self.lefteye_y[mask_trial])
                        subsets[subset_name]['ecc_L'].append(self.lefteye_ecc[mask_trial])
                        self.eye = "L"
                    if np.any(self.righteye_x):
                        subsets[subset_name]['x_pos_R'].append(self.righteye_x[mask_trial])
                        subsets[subset_name]['y_pos_R'].append(self.righteye_y[mask_trial])
                        subsets[subset_name]['ecc_R'].append(self.righteye_ecc[mask_trial])
                        self.eye = "R"
                    if (np.any(self.lefteye_x)) and (np.any(self.righteye_x)):
                        subsets[subset_name]['x_pos_bi'].append(self.bieye_x[mask_trial])
                        subsets[subset_name]['y_pos_bi'].append(self.bieye_y[mask_trial])
                        subsets[subset_name]['ecc_bi'].append(self.bieye_ecc[mask_trial])
                        self.eye = "bi"

                    subsets[subset_name]['duration'].append(np.linspace(0,0.001*len(mask_trial),len(mask_trial)))


                # Make everything the same length for calculating the mean and the median
                min_len = np.min([len(el) for el in subsets[subset_name]['x_pos_R']])

                for position,mean_position,median_position,var_position in zip(positions,mean_pos,median_pos,var_pos):

                    # Make all the subsets the same length
                    subsets[subset_name][position] = np.array([el[:min_len] for el in subsets[subset_name][position]])
                    subsets[subset_name][var_position] = np.array([np.var(trial) for trial in subsets[subset_name][position]])

                    # Calculate the median
                    subsets[subset_name][median_position] = np.median(subsets[subset_name][position],axis = 0)

                    # If the user of the function wants the median to be subtracted from the data, the function enters this statement. In that case, the median is subtracted
                    # The median is calculated over the trials. This is subtracted from each trial
                    if self.sub_med == 'True':
                        for j in range(len(subsets[subset_name][position])):
                            subsets[subset_name][position][j] = [element1 - element2 for (element1,element2) in zip(subsets[subset_name][position][j,:],subsets[subset_name][median_position])] 

                    # Calculate the mean
                    subsets[subset_name][mean_position] = np.mean(subsets[subset_name][position],axis = 0)
                
                subsets[subset_name]['duration'] = np.array([el[:min_len] for el in subsets[subset_name]['duration']])

            self.subsets = subsets

        def EHplot(self):
             
            pl.rcParams.update({'font.size': 16})
            pl.rcParams.update({'pdf.fonttype':42})
            pl.rcParams.update({'figure.max_open_warning': 0})
            pl.rcParams['axes.spines.right'] = False
            pl.rcParams['axes.spines.top'] = False 

            # Find coordinated of the stimulus
            # Load in screenshots. These are example screenshots, the size of the stimulus was not the same for all trials
            path_EH = '/home/vreugdenhil/Documents/Pilot_Data/Ebbinghaus_Screenshot/'
            image_nosur = pl.imread(path_EH +'sub-999_ses-9_task-EH_run-9_Screenshot9.png')
            image_big = pl.imread(path_EH +'sub-999_ses-9_task-EH_run-9_Screenshot7.png')

            # Find the circle for the trials without surrounding    
            white_im = np.where((image_nosur[:,:,0] == 1)) # Find where the image is white, this is the circle
            # Find the min and max values for the coordinates where the image is white
            x_min_nosur = np.min(white_im[1])
            x_max_nosur = np.max(white_im[1])
            y_min_nosur = np.min(white_im[0])
            y_max_nosur = np.max(white_im[0])

            # Find the circles for the trials with surrounding
            white_im = np.where((image_big[:,:,0] == 1))
            x_min_big = np.min(white_im[1])
            x_max_big = np.max(white_im[1])
            y_min_big = np.min(white_im[0])
            y_max_big = np.max(white_im[0])

            # Recalculate to visual angle
            x_min_nosur = (x_min_nosur - self.x_cor_fixcross)*self.deg_per_pix
            x_max_nosur = (x_max_nosur - self.x_cor_fixcross)*self.deg_per_pix
            y_min_nosur = (y_min_nosur - self.y_cor_fixcross)*self.deg_per_pix
            y_max_nosur = (y_max_nosur - self.y_cor_fixcross)*self.deg_per_pix

            x_min_big = (x_min_big - self.x_cor_fixcross)*self.deg_per_pix
            x_max_big = (x_max_big - self.x_cor_fixcross)*self.deg_per_pix
            y_min_big = (y_min_big - self.y_cor_fixcross)*self.deg_per_pix
            y_max_big = (y_max_big - self.y_cor_fixcross)*self.deg_per_pix

            # Calculate the radius of the circle
            r_stim_nosur_L = -2.5 - x_min_nosur
            r_stim_nosur_R = x_max_nosur - 2.5
            r_stim_big_L = -2.5 - x_min_big
            r_stim_big_R = x_max_big - 2.5

            # Make lists to loop over for plotting
            eye = self.eye
            pos_to_plot = [f"x_pos_{eye}",f"y_pos_{eye}",f"ecc_{eye}"]
            mean_to_plot = [f"mean_x_{eye}",f"mean_y_{eye}",f"mean_ecc_{eye}"]
            median_to_plot = [f"median_x_{eye}",f"median_y_{eye}",f"median_ecc_{eye}"]
            var_to_plot = [f"var_x_{eye}",f"var_y_{eye}",f"var_ecc_{eye}"]
            current_subset_names = ['no surr', 'large surr']

            fig,ax = pl.subplots(3,len(current_subset_names),constrained_layout = True,figsize = (15,15)) # Create figure
            position_names = ['x','y','eccentricity']
            # Plot - rows x, y and eccentricity and one column for every subset
            for i,subset in enumerate(current_subset_names): # Loop over subsets
                for k,position,var_position in zip(range(len(pos_to_plot)), pos_to_plot,var_to_plot): # Loop over positions to plot (x, y, eccentricity for selected eye)
                    for dur,j in zip(self.subsets[subset]['duration'],self.subsets[subset][position]): # Loop over trial
                        ax[k,i].plot(np.array(dur),np.array(j),linewidth = 0.2, color = 'black', alpha = 0.1)
                            
                    mean_plot, = ax[k,i].plot(np.array(self.subsets[subset]['duration'][0,:]),np.array(self.subsets[subset][mean_to_plot[k]]),linewidth = 3, color = 'red', label = 'mean')
                    # median_plot, = ax[k,i].plot(np.array(self.subsets[subset]['duration'][0,:]),np.array(self.subsets[subset][median_to_plot[k]]),linewidth = 3, color = 'blue', label = 'median')

            
                    quantile_10 = np.quantile(self.subsets[subset][var_position],0.90)
                    quantile_25 = np.quantile(self.subsets[subset][var_position],0.75)

                    var_small = np.sum((self.subsets[subset][var_position] < quantile_25))
                    var_med = np.sum((self.subsets[subset][var_position] > quantile_25) & (self.subsets[subset][var_position] < quantile_10))
                    var_large = np.sum((self.subsets[subset][var_position] > quantile_10))
                    trials = len(self.subsets[subset][var_position])


                    ax[k,i].set_title(f'{subset} - {position_names[k]} position')
                    ax[k,i].set_xlabel('Time (s)')
                    ax[k,i].set_ylabel('Visual angle (degrees)')
                    ax[k,i].set_xlim([0,np.max(self.subsets[subset]['duration'])])
                    ax[k,i].text(0.1,2.75,f'median = {np.mean(np.array(self.subsets[subset][median_to_plot[k]])):.4f}')
                    
                    # Set limits of the y axis. If the median is subtracted, the eccentricity can be negative.
                    # If the median is not subtracted, the eccentricity cannot be negative.
                    # Therefore, the y-axis goes to negative values if the median is subtracted, but not if it is not subtracted. 
                    if k != 2:
                        ax[k,i].set_ylim([-3,3])
                    elif k == 2:
                        if self.sub_med == 'True':
                            ax[k,i].set_ylim([-3,3])
                        elif self.sub_med == 'False':
                            ax[k,i].set_ylim([0,3])


                # Plot coordinates of the stimulus
                if subset == 'large surr':
                    stim_cen = ax[0,i].axhline(-2.5,linewidth = 1,linestyle = '--',label = 'stimulus center') 
                    stim_cen = ax[0,i].axhline(2.5,linewidth = 1,linestyle = '--') 
                    stim_ext = ax[0,i].axhline(-2.5 - r_stim_big_L,linewidth = 1,linestyle = '--',alpha = 0.8,label = 'stimulus extent') 
                    stim_ext = ax[0,i].axhline(-2.5 + r_stim_big_L,linewidth = 1,linestyle = '--',alpha = 0.8)
                    stim_ext = ax[0,i].axhline(2.5 - r_stim_big_R,linewidth = 1,linestyle = '--',alpha = 0.8) 
                    stim_ext = ax[0,i].axhline(2.5 + r_stim_big_R,linewidth = 1,linestyle = '--',alpha = 0.8)

                    stim_cen = ax[1,i].axhline(0,linewidth = 1,linestyle = '--',label = 'stimulus center') 
                    stim_cen = ax[1,i].axhline(0,linewidth = 1,linestyle = '--') 
                    stim_ext = ax[1,i].axhline(0 - r_stim_big_L,linewidth = 1,linestyle = '--',alpha = 0.8,label = 'stimulus extent') 
                    stim_ext = ax[1,i].axhline(0 + r_stim_big_L,linewidth = 1,linestyle = '--',alpha = 0.8) 
                    stim_ext = ax[1,i].axhline(0 - r_stim_big_R,linewidth = 1,linestyle = '--',alpha = 0.8) 
                    stim_ext = ax[1,i].axhline(0 + r_stim_big_R,linewidth = 1,linestyle = '--',alpha = 0.8) 

                    stim_cen = ax[2,i].axhline(-2.5,linewidth = 1,linestyle = '--',label = 'stimulus center') 
                    stim_cen = ax[2,i].axhline(2.5,linewidth = 1,linestyle = '--') 
                    stim_ext = ax[2,i].axhline(-2.5 - r_stim_big_L,linewidth = 1,linestyle = '--',alpha = 0.8, label = 'stimulus extent') 
                    stim_ext = ax[2,i].axhline(-2.5 + r_stim_big_L,linewidth = 1,linestyle = '--',alpha = 0.8)
                    stim_ext = ax[2,i].axhline(2.5 - r_stim_big_R,linewidth = 1,linestyle = '--',alpha = 0.8) 
                    stim_ext = ax[2,i].axhline(2.5 + r_stim_big_R,linewidth = 1,linestyle = '--',alpha = 0.8)

                # Lines for the no surround stimuli
                if subset == 'no surr':
                    stim_cen = ax[0,i].axhline(-2.5,linewidth = 1,linestyle = '--',label = 'stimulus center') 
                    stim_cen = ax[0,i].axhline(2.5,linewidth = 1,linestyle = '--') 
                    stim_ext = ax[0,i].axhline(-2.5 - r_stim_nosur_L,linewidth = 1,linestyle = '--',alpha = 0.8,label = 'stimulus extent') 
                    stim_ext = ax[0,i].axhline(-2.5 + r_stim_nosur_L,linewidth = 1,linestyle = '--',alpha = 0.8)
                    stim_ext = ax[0,i].axhline(2.5 - r_stim_nosur_R,linewidth = 1,linestyle = '--',alpha = 0.8) 
                    stim_ext = ax[0,i].axhline(2.5 + r_stim_nosur_R,linewidth = 1,linestyle = '--',alpha = 0.8)

                    stim_cen = ax[1,i].axhline(0,linewidth = 1,linestyle = '--',label = 'stimulus center') 
                    stim_cen = ax[1,i].axhline(0,linewidth = 1,linestyle = '--') 
                    stim_ext = ax[1,i].axhline(0 - r_stim_nosur_L,linewidth = 1,linestyle = '--',alpha = 0.8,label = 'stimulus extent') 
                    stim_ext = ax[1,i].axhline(0 + r_stim_nosur_L,linewidth = 1,linestyle = '--',alpha = 0.8) 
                    stim_ext = ax[1,i].axhline(0 - r_stim_nosur_R,linewidth = 1,linestyle = '--',alpha = 0.8) 
                    stim_ext = ax[1,i].axhline(0 + r_stim_nosur_R,linewidth = 1,linestyle = '--',alpha = 0.8) 

                    stim_cen = ax[2,i].axhline(-2.5,linewidth = 1,linestyle = '--',label = 'stimulus center') 
                    stim_cen = ax[2,i].axhline(2.5,linewidth = 1,linestyle = '--') 
                    stim_ext = ax[2,i].axhline(-2.5 - r_stim_nosur_L,linewidth = 1,linestyle = '--',alpha = 0.8,label = 'stimulus extent') 
                    stim_ext = ax[2,i].axhline(-2.5 + r_stim_nosur_L,linewidth = 1,linestyle = '--',alpha = 0.8)
                    stim_ext = ax[2,i].axhline(2.5 - r_stim_nosur_R,linewidth = 1,linestyle = '--',alpha = 0.8) 
                    stim_ext = ax[2,i].axhline(2.5 + r_stim_nosur_R,linewidth = 1,linestyle = '--',alpha = 0.8)
    
            
            pl.suptitle(f"Gaze task: EH {self.expsettings['sub']} {self.expsettings['ses']} {self.expsettings['run']} eye: {eye}",fontsize = 24)
            
            fig.legend([mean_plot],['mean'],loc = 'center right',bbox_to_anchor = (1.2,0.5)) 
            fig.savefig(self.out_path + f"/{self.expsettings['sub']}_{self.expsettings['ses']}_{self.expsettings['run']}_{self.expsettings['task']}_gaze.pdf", dpi=600, bbox_inches='tight', transparent=True)

            return
        
        def CSsubset(self):

            # Get the onsets and phases from the TSV file
            onsets = self.eventsfile['onset'] # Onsets of the events
            phases = self.eventsfile['phase'] # Which phase
            event = self.eventsfile['event_type'] # Is a stimulus shown?
            frames = self.eventsfile['nr_frames'] # Frames shown in experiment

           # frames = list(frames)#all trials
            ind = np.where(((phases == 0) & (event == 'stim') & (frames > 0))) # Stimulus onset
            ind = ind[0]
            if frames.iloc[-1] > 0:
                ind = ind[:-1]
            ind_offset = ind + 5 # Stimulus offset
            onsets_stim_s =  np.array(onsets[ind])
            offsets_stim_s =  np.array(onsets[ind_offset])

            # read in relevant column to subset trials with or without surround
            sur_tsv = self.eventsfile['target_surround_presence']

            # only trials with surround
            sur = np.where((sur_tsv == True) & (phases == 0) & (event == 'stim') & (frames > 0))
            sur = sur[0]
            if frames.iloc[-1] > 0:
                sur = sur[:-1]
            sur_off = sur + 5
            time_sur = np.array(onsets[sur])
            time_sur_off = np.array(onsets[sur_off])

            # no surround
            no_sur = np.where((sur_tsv == False) & (phases == 0) & (event == 'stim') & (frames > 0))
            no_sur = no_sur[0]
            if frames.iloc[-1] > 0:
                no_sur = no_sur[:-1]
            no_sur_off = no_sur + 5
            time_no_sur = np.array(onsets[no_sur])
            time_no_sur_off = np.array(onsets[no_sur_off])

            # Define onset and offset times for each subset, this is used later to create a mask to select the data for each subset
            time_subsets_on = dict() # Onset times
            time_subsets_off = dict() # Offset times

            # Subsets
            all_subset_names = ['no surr','surr'] # All the subsets from the data
            current_subset_names = ['no surr', 'surr'] # The subsets you want to plot

            # Positions
            positions = ['x_pos_L','y_pos_L','ecc_L','x_pos_R','y_pos_R','ecc_R','x_pos_bi','y_pos_bi','ecc_bi']
            mean_pos = ['mean_x_L','mean_y_L','mean_ecc_L','mean_x_R','mean_y_R','mean_ecc_R','mean_x_bi','mean_y_bi','mean_ecc_bi']
            median_pos = ['median_x_L','median_y_L','median_ecc_L','median_x_R','median_y_R','median_ecc_R','median_x_bi','median_y_bi','median_ecc_bi']
            var_pos = ['var_x_L','var_y_L','var_ecc_L','var_x_R','var_y_R','var_ecc_R','var_x_bi','var_y_bi','var_ecc_bi']

            # From which eye would you want to plot the data? left, right or binocular
        
            time_subsets_on = dict()
            time_subsets_off = dict()

            for subset_name in all_subset_names: # Onset and offset times for surrounding and no surrounding

                if subset_name == 'no surr':
                    time_subsets_on[subset_name] = time_no_sur
                    time_subsets_off[subset_name] = time_no_sur_off

                elif subset_name == 'surr':
                    time_subsets_on[subset_name] = time_sur
                    time_subsets_off[subset_name] = time_sur_off
                    

            subsets = dd(lambda:dd(list))

            for subset_name in current_subset_names:

                time_subset_on = time_subsets_on[subset_name] # Onset times
                time_subset_off = time_subsets_off[subset_name] # Offset times


                for i in range(len(time_subset_on)):

                    mask_trial = (self.time>=time_subset_on[i]) & (self.time<time_subset_off[i]) # mask to select data for each subset

                    # Select data for each subset
                    if np.any(self.lefteye_x):
                        subsets[subset_name]['x_pos_L'].append(self.lefteye_x[mask_trial])
                        subsets[subset_name]['y_pos_L'].append(self.lefteye_y[mask_trial])
                        subsets[subset_name]['ecc_L'].append(self.lefteye_ecc[mask_trial])
                        self.eye = "L"
                    if np.any(self.righteye_x):
                        subsets[subset_name]['x_pos_R'].append(self.righteye_x[mask_trial])
                        subsets[subset_name]['y_pos_R'].append(self.righteye_y[mask_trial])
                        subsets[subset_name]['ecc_R'].append(self.righteye_ecc[mask_trial])
                        self.eye = "R"
                    if (np.any(self.lefteye_x)) and (np.any(self.righteye_x)):
                        subsets[subset_name]['x_pos_bi'].append(self.bieye_x[mask_trial])
                        subsets[subset_name]['y_pos_bi'].append(self.bieye_y[mask_trial])
                        subsets[subset_name]['ecc_bi'].append(self.bieye_ecc[mask_trial])
                        self.eye = "bi"

                    subsets[subset_name]['duration'].append(np.linspace(0,0.001*len(mask_trial),len(mask_trial)))


                # Make everything the same length for calculating the mean and the median
                min_len = np.min([len(el) for el in subsets[subset_name]['x_pos_R']])

                for position,mean_position,median_position,var_position in zip(positions,mean_pos,median_pos,var_pos):

                    # Make all the subsets the same length
                    subsets[subset_name][position] = np.array([el[:min_len] for el in subsets[subset_name][position]])
                    subsets[subset_name][var_position] = np.array([np.var(trial) for trial in subsets[subset_name][position]])

                    subsets[subset_name][median_position] = np.median(subsets[subset_name][position],axis = 0)

                    # If the user of the function wants the median to be subtracted from the data, the function enters this statement. In that case, the median is subtracted
                    # The median is calculated over the trials. This is subtracted from each trial
                    if self.sub_med == 'True':
                        for j in range(len(subsets[subset_name][position])):
                            subsets[subset_name][position][j] = [element1 - element2 for (element1,element2) in zip(subsets[subset_name][position][j,:],subsets[subset_name][median_position])] 
                
                    subsets[subset_name][mean_position] = np.mean(subsets[subset_name][position],axis = 0)
                subsets[subset_name]['duration'] = np.array([el[:min_len] for el in subsets[subset_name]['duration']])

            self.subsets = subsets

        def CSplot(self):
            
            pl.rcParams.update({'font.size': 16})
            pl.rcParams.update({'pdf.fonttype':42})
            pl.rcParams.update({'figure.max_open_warning': 0})
            pl.rcParams['axes.spines.right'] = False
            pl.rcParams['axes.spines.top'] = False 

            # Create lists to loop over for plotting
            eye = self.eye # Select eye
            pos_to_plot = [f"x_pos_{eye}",f"y_pos_{eye}",f"ecc_{eye}"]
            mean_to_plot = [f"mean_x_{eye}",f"mean_y_{eye}",f"mean_ecc_{eye}"]
            median_to_plot = [f"median_x_{eye}",f"median_y_{eye}",f"median_ecc_{eye}"]
            var_to_plot = [f"var_x_{eye}",f"var_y_{eye}",f"var_ecc_{eye}"]
            current_subset_names = ['no surr', 'surr']

            fig,ax = pl.subplots(3,len(current_subset_names),constrained_layout = True,figsize = (15,15)) # Create figure
            position_names = ['x','y','eccentricity']
            # Plot - rows x, y and eccentricity and one column for every subset
            for i,subset in enumerate(current_subset_names): # Loop over subsets
                for k,position,var_position in zip(range(len(pos_to_plot)), pos_to_plot,var_to_plot): # Loop over positions to plot (x, y, eccentricity for selected eye)
                    for dur,j in zip(self.subsets[subset]['duration'],self.subsets[subset][position]): # Loop over trials     
                        ax[k,i].plot(np.array(dur),np.array(j),linewidth = 0.2, color = 'black', alpha = 0.1)
                            

                    mean_plot, = ax[k,i].plot(np.array(self.subsets[subset]['duration'][0,:]),np.array(self.subsets[subset][mean_to_plot[k]]),linewidth = 3, color = 'red', label = 'mean')
                    # median_plot, = ax[k,i].plot(np.array(self.subsets[subset]['duration'][0,:]),np.array(self.subsets[subset][median_to_plot[k]]),linewidth = 3, color = 'blue', label = 'median')

                    quantile_10 = np.quantile(self.subsets[subset][var_position],0.90)
                    quantile_25 = np.quantile(self.subsets[subset][var_position],0.75)

                    var_small = np.sum((self.subsets[subset][var_position] < quantile_25))
                    var_med = np.sum((self.subsets[subset][var_position] > quantile_25) & (self.subsets[subset][var_position] < quantile_10))
                    var_large = np.sum((self.subsets[subset][var_position] > quantile_10))
                    trials = len(self.subsets[subset][var_position])


                    ax[k,i].set_title(f'{subset} - {position_names[k]} position')
                    ax[k,i].set_xlabel('Time (s)')
                    ax[k,i].set_ylabel('Visual angle (degrees)')
                    ax[k,i].set_xlim([0,np.max(self.subsets[subset]['duration'])])
                    ax[k,i].text(0.1,3.75,f'median = {np.mean(np.array(self.subsets[subset][median_to_plot[k]])):.4f}'), 
                    
                    # Set limits of the y axis. If the median is subtracted, the eccentricity can be negative.
                    # If the median is not subtracted, the eccentricity cannot be negative.
                    # Therefore, the y-axis goes to negative values if the median is subtracted, but not if it is not subtracted. 
                    if k != 2:
                        ax[k,i].set_ylim([-4,4])
                    elif k == 2:
                        if self.sub_med == 'True':
                            ax[k,i].set_ylim([-4,4])
                        elif self.sub_med == 'False':
                            ax[k,i].set_ylim([0,4])
            

                # x-coordinate
                for coordinate in range(2): 
                    # Lines for the large surround stimuli
                    # X
                    stim_cen = ax[0,i].axhline(0,linewidth = 1,linestyle = '--',label = 'stimulus center') 
                    # y
                    stim_cen = ax[1,i].axhline(0,linewidth = 1,linestyle = '--',label = 'stimulus center') 
                    
            pl.suptitle(f"Gaze task: CS {self.expsettings['sub']} {self.expsettings['ses']} {self.expsettings['run']} eye: {eye}",fontsize = 24)
            
            fig.legend([mean_plot],['mean'],loc = 'center right',bbox_to_anchor = (1.2,0.5)) 
            fig.savefig(self.out_path + f"/{self.expsettings['sub']}_{self.expsettings['ses']}_{self.expsettings['run']}_{self.expsettings['task']}_gaze.pdf", dpi=600, bbox_inches='tight', transparent=True)

            return
        
        def pRFsubset(self):
            
            # Load in screenshots of the task, we use this to detect the trials with and without bar. 
            screenshots_folder = '/data1/projects/dumoulinlab/Lab_members/Marco/SM-pRF/sourcedata/sub-001/ses-1/task/sub-001_ses-1_task-2R_run-1_Logs/sub-001_ses-1_task-2R_run-1_Screenshots'
            filenames_list = os.listdir(screenshots_folder)
            filenames_ordered = sorted(filenames_list)
            
            # Read in imaged
            image_list = []    
            for filename in filenames_ordered: 
                im=pl.imread(screenshots_folder +'/'+ filename)
                image_list.append(im)

            # Select an image without bar
            nobar = image_list[0]
            # test_bar = self.barscreen[100]

            bar_on_screen = []
            bar_not_screen = []

            # Images with bar, have more colour values than the ones without bar. So the list of colour values is longer
            # Make a list with True and False for when the bar was on screen and not, we use this later to index in the trials
            for i,image in enumerate(image_list):
                if len(np.unique(image)) > len(np.unique(nobar)):
                    bar_on_screen.append(True)
                    bar_not_screen.append(False)
                else:
                    bar_on_screen.append(False)
                    bar_not_screen.append(True)

            bar_on_screen.append(False)
            bar_not_screen.append(False)

            # Find the onsets of the TRs
            ind_TR = self.eventsfile['response'] == 't'
            onsets_TR = np.array(self.eventsfile['onset'][ind_TR]) 
            
            # Make everything the same length
            onsets_TR = onsets_TR[:len(bar_on_screen)]
            if len(onsets_TR) < len(bar_on_screen):
                bar_on_screen = bar_on_screen[:len(onsets_TR)]
            onsets_bar = onsets_TR[bar_on_screen] # find onsets of the trials when the bar was on the screen
            if len(onsets_TR) < len(bar_not_screen):
                bar_not_screen = bar_not_screen[:len(onsets_TR)]
            onsets_nobar = onsets_TR[bar_not_screen] # Find onsets of the trials when the bar was not on the screen
            bar_passes = np.zeros((8,20))
            bar_interval = {}
            start = 0
            end = 20 # Barpass is 20 TRs

            # Put the trials in an arrays
            for i in range(len(bar_passes)):
                bar_passes[i,:] = onsets_bar[start:end] # Find the times for the trials
                start = start + 20
                end = end + 20
            start = 0
            end = 20
            for i in range(6):
                if i == 0:
                    bar_interval[i] = onsets_nobar[start:end]
                    start = 20
                    end = start + 15 # Interval is 15 TRs
                else:
                    bar_interval[i] = onsets_nobar[start:end]
                    start = start + 15
                    end = end + 15

                

            if np.any(self.lefteye_x):
                x_pRF = self.lefteye_x
                y_pRF = self.lefteye_y
                ecc_pRF = self.lefteye_ecc
                self.eye = "L"
            if np.any(self.righteye_x):
                x_pRF = self.righteye_x
                y_pRF = self.righteye_y
                ecc_pRF = self.righteye_ecc
                self.eye = "R"
            if (np.any(self.lefteye_x)) and (np.any(self.righteye_x)):
                x_pRF = self.bieye_x
                y_pRF = self.bieye_y
                ecc_pRF = self.bieye_ecc
                self.eye = "bi"

           
            bar_pass_dict = dd(lambda:dd(list))

            for i in range(len(bar_passes)):
                mask_barpass = (self.time>=bar_passes[i,0]) & (self.time<bar_passes[i,19]) # mask to select data for the barpass trials

                # Append data for each trial with barpass to a dictionairy
                bar_pass_dict['bar pass']['x_pos'].append(x_pRF[mask_barpass])
                bar_pass_dict['bar pass']['y_pos'].append(y_pRF[mask_barpass])
                bar_pass_dict['bar pass']['eccentricity'].append(ecc_pRF[mask_barpass])

            for i in range(len(bar_interval)):
                mask_nobar = (self.time>=bar_interval[i][0]) & (self.time<bar_interval[i][-1]) # Mask to select data for the no bar trials
                
                # Append data for each trial without bar to a dictionairy
                bar_pass_dict['no bar pass']['x_pos'].append(x_pRF[mask_nobar])
                bar_pass_dict['no bar pass']['y_pos'].append(y_pRF[mask_nobar])
                bar_pass_dict['no bar pass']['eccentricity'].append(ecc_pRF[mask_nobar])   


            # Compute mean & median
            min_len = np.min([len(el) for el in bar_pass_dict['bar pass']['x_pos']]) # Make arrays the same size
            min_len_no = np.min([len(el) for el in bar_pass_dict['no bar pass']['x_pos']])
            # Create lists to loop over
            positions = ['x_pos','y_pos','eccentricity']
            mean_position = ['x_pos_mean','y_pos_mean','eccentricity_mean']
            median_position = ['x_pos_median','y_pos_median','eccentricity_median']
            var_position = ['x_pos_var','y_pos_var','eccentricity_var']
            for position,mean_pos,median_pos,var_pos in zip(positions,mean_position,median_position,var_position):

                # Barpass

                bar_pass_dict['bar pass'][position] = np.array([el[:min_len] for el in bar_pass_dict['bar pass'][position]]) # Make all arrays the same length
                bar_pass_dict['bar pass'][median_pos] = np.median(bar_pass_dict['bar pass'][position],axis = 0) # Calculate the median
                bar_pass_dict['bar pass'][var_pos] = np.array([np.var(trial) for trial in bar_pass_dict['bar pass'][position]]) # Compute the variance for each trial
               
                # If the user of the function wants the median to be subtracted from the data, the function enters this statement. In that case, the median is subtracted
                # The median is calculated over the trials. This is subtracted from each trial
                if self.sub_med == 'True':
                    for j in range(len(bar_pass_dict['bar pass'][position])):
                        bar_pass_dict['bar pass'][position][j] = [element1 - element2 for (element1,element2) in zip(bar_pass_dict['bar pass'][position][j,:],bar_pass_dict['bar pass'][median_pos])]

                bar_pass_dict['bar pass'][mean_pos] = np.mean(bar_pass_dict['bar pass'][position],axis = 0) # Calculate the mean

                # No barpass

                bar_pass_dict['no bar pass'][position] = np.array([el[:min_len_no] for el in bar_pass_dict['no bar pass'][position]]) # Make all arrays the same size
                bar_pass_dict['no bar pass'][var_pos] = np.array([np.var(trial) for trial in bar_pass_dict['no bar pass'][position]]) # Calculate the variance for each trial
                bar_pass_dict['no bar pass'][median_pos] = np.median(bar_pass_dict['no bar pass'][position],axis = 0) # Calculate the median 

                # If the user of the function wants the median to be subtracted from the data, the function enters this statement. In that case, the median is subtracted
                # The median is calculated over the trials. This is subtracted from each trial
                if self.sub_med == 'True':
                    for j in range(len(bar_pass_dict['no bar pass'][position])):
                        bar_pass_dict['no bar pass'][position][j] = [element1 - element2 for (element1,element2) in zip(bar_pass_dict['no bar pass'][position][j,:],bar_pass_dict['no bar pass'][median_pos])]

                bar_pass_dict['no bar pass'][mean_pos] = np.mean(bar_pass_dict['no bar pass'][position],axis = 0) # calculate the mean
               
            bar_pass_dict['bar pass']['duration'].append(np.linspace(0,0.001*min_len,min_len)) # Create list with the time in seconds, to use on the x-axis
            bar_pass_dict['no bar pass']['duration'].append(np.linspace(0,0.001*min_len_no,min_len_no))

            self.bar_pass_dict = bar_pass_dict

            return

        def pRFplot(self):
            if np.any(self.lefteye_x):
                self.eye = "L"
            if np.any(self.righteye_x):
                self.eye = "R"
            if (np.any(self.lefteye_x)) and (np.any(self.righteye_x)):
                self.eye = "bi"


            

            # Make lists to loop over for plotting
            sub_barpass = ['bar pass','no bar pass']

            median_position = ['x_pos_median','y_pos_median','eccentricity_median']
            var_position = ['x_pos_var','y_pos_var','eccentricity_var']
            pos_to_plot = ['x_pos','y_pos','eccentricity']

            fig,ax = pl.subplots(3,len(sub_barpass),constrained_layout = True,figsize = (30,15)) # Create figure
            
            # plotting pRF
            for k,subset in enumerate(sub_barpass):   
                for i,position in enumerate(pos_to_plot):
                    for j in range(len((self.bar_pass_dict[subset]['eccentricity']))):
                        # plot a line for each bar pass or interval, for each position
                        ax[i,k].plot(np.array(self.bar_pass_dict[subset]['duration'][0]),np.array(self.bar_pass_dict[subset][position][j,:]),linewidth = 0.8, color = 'black', alpha = 0.3)
                   
                    # Plot the mean of the trials
                    # mean_plot, = ax[i,k].plot(np.array(self.bar_pass_dict[subset]['duration'][0]),np.array(self.bar_pass_dict[subset][mean_position[i]]),linewidth = 1.5, color = 'red', label = 'mean')
                    # median_plot, = ax[i,k].plot(np.array(self.bar_pass_dict[subset]['duration'][0]),np.array(self.bar_pass_dict[subset][median_position[i]]),linewidth = 1.5, color = 'blue', label = 'median')

                
                    mean_var = np.mean(self.bar_pass_dict[subset][var_position[i]]) # Compute the mean of the variances
                    trials = len(self.bar_pass_dict[subset]['x_pos_var']) # Total amount of trials

                    ax[i,k].set_title(f'{subset} - {position}')
                    ax[i,k].set_xlabel('Time (s)')
                    ax[i,k].set_ylabel('Visual angle (degrees)')
                    ax[i,k].text(0.1,3.5,f'median = {np.mean(np.array(self.bar_pass_dict[subset][median_position[i]])):.4f}')
             

                    if i != 2:
                        ax[i,k].set_ylim([-4,4])
                    elif i == 2:
                        ax[i,k].set_ylim([-4,4])       

            pl.suptitle(f"Gaze - task: pRF per bar pass {self.expsettings['sub']} {self.expsettings['ses']} {self.expsettings['run']} eye: {self.eye}",fontsize = 24)
            # fig.legend([mean_plot],['mean'],loc = 'center right',bbox_to_anchor = (1.2,0.5)) 

            fig.savefig(self.out_path + f"/{self.expsettings['sub']}_{self.expsettings['ses']}_{self.expsettings['run']}_{self.expsettings['task']}_gazebarpass.pdf", dpi=600, bbox_inches='tight', transparent=True)
            
            return


                
        def PupilPlot(self):
            # Plots the pupil size over time for each task. 
            fig = pl.figure(constrained_layout = True, figsize = (15,15))
            if np.any(self.pupil_left):
                Pupil = self.pupil_left
                eye = 'left'
            elif np.any(self.pupil_right):
                Pupil = self.pupil_right
                eye = 'right'
            elif (np.any(self.pupil_left)) & (np.any(self.pupil_right)):
                Pupil = np.mean([self.pupil_left, self.pupil_right],axis = 0)
                eye = 'both'

            mean_pupil = np.mean(Pupil)
            pl.plot(self.time,Pupil)
            pl.ylim([0,6000])
            pl.xlabel('Time (s)')
            pl.ylabel('Pupil size')
            pl.text(250,5000,f'mean pupil size = {mean_pupil}')
           
            
            pl.title(f"Pupil size task: {self.expsettings['task']} sub: {self.expsettings['sub']} ses: {self.expsettings['ses']} run: {self.expsettings['run']} eye: {eye}",fontsize = 24)
            fig.savefig(self.out_path + f"/{self.expsettings['sub']}_{self.expsettings['ses']}_{self.expsettings['run']}_{self.expsettings['task']}_pupil.pdf", dpi=600, bbox_inches='tight', transparent=True)
            pickle.dump(Pupil, open('/home/vreugdenhil/Documents/test_pupilsize/' + f"pupilsize_10mg_sub-{self.expsettings['sub']}_task-{self.expsettings['task']}_run-{self.expsettings['run']}.pkl", 'wb'))
            
            return