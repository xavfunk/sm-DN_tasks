"""
Visual detection task

• Gabor stimuli are presented on 50% of all trials, hidden in continously presented dynamic noise.
• Participants are asked to press either of the 4 buttons indicating whether
  they saw a Gabor patch or not and whether they are confident about their
  choice or not.

• Participants will have to be instructed to fairly equally distribute their confidence reports. 
  The task is quite challenging, but they have to refrain from saying low confidence all the time. 

• Because of the continuous noise, the onset of the stimulus interval is cued by a change in color of the
  fixation dot (to green). The dot remains green during the response interval and will turn black after the response.

• If participants did not press a button on time (1.4s after stimulus onset) the fixation dot will 
  momentarily turn white. 

• There is no other trial-to-trial feedback, but participants see the score + average confidence of
  the previous block during blocks breaks. 

• The settings of the original experiment are 4 blocks of 100 trials. The experiment should not last longer than 20 minutes.

to check GPU usage: watch -n 2 nvidia-smi
"""

import os
import numpy as np
import pandas as pd
import glob, shutil
import sys
import random
import os.path as op

import psychopy
from psychopy import logging, visual, event
from psychopy.visual import GratingStim, filters, Rect, Circle, ImageStim

logging.console.setLevel(logging.WARNING)

from exptools2.core.trial import Trial
from exptools2.core.eyetracker import PylinkEyetrackerSession
from IPython import embed as shell
import matplotlib.pyplot as plt
from PIL import Image
# from utils import quick_fit, gaussian



# np.random.seed(1)   # NOTE: because of this line all experiments will always be idential in the order of stimuli. Feel free to remove if unwanted.

class DetectTrial(Trial):
    def __init__(self, parameters = {}, phase_names =[], phase_durations=[], session=None, monitor=None, tracker=None, ID=0, intensity=1): # self, parameters = {}, phase_durations=[], session=None, monitor=None, tracker=None, ID=0,
        self.monitor = monitor
        self.parameters = parameters
        self.ID = ID
        self.phase_durations = phase_durations  
        self.session = session
        self.miniblock = np.floor(self.ID/self.session.block_len)
        self.phase_names = phase_names  
        self.noise_played = False
        self.signal_tone_played = False
        self.intensity = intensity
        
        self.create_stimuli()
        self.parameters.update({'response': -1, 
                                'correct': -1,
                                'miniblock': self.miniblock ,
                                'blinkDuringTrial': 0,
                                'RT': -1,
                                'trial': self.ID,
                                'confidence': -1})        

        self.stopped = False
        super(DetectTrial, self).__init__(phase_durations = phase_durations,
                                         parameters = parameters,
                                         phase_names = self.phase_names,  
                                         session = self.session, 
                                         trial_nr = self.ID)


    def create_stimuli(self):

        # self.stim_sizePIX = np.round(signal_parameters['size'] * self.session.pix_per_deg).astype(int)
        # self.noise_sizePIX = np.round(noise_parameters['size'] * self.session.pix_per_deg / 2).astype(int) # <= notice division by 2, it's to increase the noise element size to 2 later
        
        self.grating = self.session.stimuli[self.parameters['signal_opacity']]
        # print(f'selected {self.grating}' )

        # take from pre-made stimuli at session level
        self.rand_id = np.random.randint(0,self.session.n_noise_textures)
        self.previous_rand_id = self.rand_id
        self.noise = self.session.noise_stimuli[self.rand_id]

        self.stim_set = False # little helping flag to set stimuli params once in draw()

        if (self.ID % self.session.block_len == 0) and self.ID>0:
            # TODO fix the feedback
            # conf = int(np.array(self.session.confidence)[-self.session.block_len:][np.array(self.session.confidence)[-self.session.block_len:] >= 0].sum() / float(self.session.block_len) * 100.0)
            # perf = int(np.array(self.session.corrects)[-self.session.block_len:][np.array(self.session.corrects)[-self.session.block_len:] >= 0].sum() / float(self.session.block_len) * 100.0)
            # intro_text = f"""You had {perf}% of correct responses and {conf}% trials with high confidence.
            # Press the spacebar to continue."""

            # pause_text = f"""Break.\nPress the central button to continue."""

            pause_text = f"Great, you did {self.ID} of {self.session.max_trials} trials.\nYou can rest a little now, but try to keep your head stable.\nContinue with any button if you're ready."
            self.pause_message = visual.TextStim(self.session.win, pos=[0,0], text= pause_text, color = (1.0, 1.0, 1.0), height=0.5, font='Arial', wrapWidth=850)

        else:
            pause_text = ''

        self.message = visual.TextStim(self.session.win, pos=[0,0], text= pause_text, color = (1.0, 1.0, 1.0), height=0.5, font='Arial', wrapWidth=850)
        
        if self.ID == 0:
            if self.session.settings['response']['metacognition'] == 'together':
                self.intro = ImageStim(self.session.win, 'assets/instructions_screen_SigDet.png', units='pix', size = [int(1920 * .95), int(1080*.95)])
            else:
                self.intro = ImageStim(self.session.win, 'assets/instructions_screen_SigDet_seperate_final.png', units='pix', size = [int(1920 * .95), int(1080*.95)])


        # making sure trial 1 is visible for movie
        if (self.ID == 1) and self.session.settings['stimuli']['screenshot']:
            self.grating.setOpacity(.1)

    def draw(self):
        if self.phase != 0:
            # update noise pattern on refresh frames
            if self.session.nr_frames_no_reset % self.session.noise_parameters['refresh_frame'] == 0:
                # noiseTexture = np.random.random([self.noise_sizePIX,self.noise_sizePIX])*2.-1.
                # noiseTexture = np.repeat(np.repeat(noiseTexture, 2, axis=1), 2, axis=0)
                
                # select from preloaded noise
                # noiseTexture = self.session.noise_textures[np.random.randint(0,100)]
                # self.noise.tex = noiseTexture

                # select from pre-made stimuli at session level
                self.rand_id = np.random.randint(0,self.session.n_noise_textures)
                if self.rand_id == self.previous_rand_id: 
                    self.rand_id += 1

                self.previous_rand_id = self.rand_id
                self.noise = self.session.noise_stimuli[self.rand_id%self.session.n_noise_textures]
                # apply rotation
                self.session.noise_stimuli[self.rand_id%self.session.n_noise_textures].ori += 90

        if self.phase == 0: # Block start + instructions
            
            if self.ID == 0: 
                
                # draw intro screen
                self.intro.draw()

                # draw stimulus check if desired
                # self.stimulus_check()

            elif self.ID % self.session.block_len == 0:
                self.message.draw()

        elif self.phase == 1: # baseline
            # draw:
            self.noise.draw()
            self.session.white_fix.draw()

        elif self.phase == 2: # stimulus interval
            self.noise.draw()                 

            if self.parameters['signal_present']:
                self.grating.draw()      

            if (self.ID == 1) and self.session.settings['stimuli']['screenshot']:
                self.grating.draw()           

            # self.session.green_fix.draw()
            self.session.white_fix.draw()

        elif self.phase == 3: # Response interval
            self.noise.draw()
            self.session.green_fix.draw()

        elif self.phase == 4:  # Feedback <- only if someone did not press a button

            self.noise.draw()
            if self.parameters['response'] == -1:
                # self.fixation.color = 'white'
                self.session.black_fix.draw()

            else:
                if self.session.settings['response']['advance_directly']:
                    # advancing directly if response was correct
                    self.stop_phase()
                else:
                    self.session.white_fix.draw()

        elif self.phase == 5: 

            if self.session.settings['response']['metacognition'] == 'together':
                # ITI
                self.noise.draw()
                self.session.white_fix.draw()
                
            elif self.session.settings['response']['metacognition'] == 'split':
                # metacog response
                self.noise.draw()
                self.session.blue_fix.draw()

        elif self.phase == 6: # metacog feedback
            self.noise.draw()
            
            if self.parameters['confidence'] == -1:
                # self.fixation.color = 'white'
                self.session.black_fix.draw()

            else:
                if self.session.settings['response']['advance_directly']:
                    # advancing directly if response was correct
                    self.stop_phase()
                else:
                    self.session.white_fix.draw()

        elif self.phase == 7: # split metacog ITI
            self.noise.draw()
            self.session.white_fix.draw()

        if (self.ID == 1) and self.session.settings['stimuli']['screenshot']:
                self.session.win.getMovieFrame()

    def event(self):
        # a-1 s-3 k-4 l-2 keyboard-button box
        # trigger = None
        for ev, t in event.getKeys(timeStamped=self.session.clock):
           
            if len(ev) > 0:
                print(self.ID)
                if self.ID < 0: # tutorial trial
                    print(ev)
                    self.stop_phase()
            
                if self.session.settings['response']['metacognition'] == 'together':
                
                    if ev in ['esc', 'escape', 'q']:
                        self.stopped = True
                        self.session.stopped = True

                        print('run canceled by user')
                        self.stop_trial()
                        if self.phase == 0:
                            self.first_trial_hold = False
                            self.stop_phase()
                    

                    elif (ev == 'space') or (ev == 'return'):
                        if self.phase == 0:
                            self.first_trial_hold = False
                            self.stop_phase()

                    elif ev == self.session.response_button_mapping['absent_confident']:

                        if self.phase == 3:
                            idx = self.session.global_log.shape[0]
                            stim_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.ID) & \
                                                                    (self.session.global_log['event_type']=='stimulus_window'), 'onset'].to_numpy()

                            RT = t - stim_onset

                            self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                            self.session.global_log.loc[idx, 'onset'] = t
                            self.session.global_log.loc[idx, 'event_type'] = 'response'
                            self.session.global_log.loc[idx, 'phase'] = self.phase
                            self.session.global_log.loc[idx, 'key'] = ev
                            self.session.global_log.loc[idx, 'nr_frames'] = self.session.nr_frames 
                            self.session.global_log.loc[idx, 'response'] = 0
                            self.session.global_log.loc[idx, 'confidence'] = 1
                            self.session.global_log.loc[idx, 'RT'] = RT

                            self.parameters['response'] = 0
                            self.parameters['confidence'] = 1
                            self.parameters['RT'] = RT

                            if self.parameters['signal_present'] ==  self.session.global_log.loc[idx, 'response']:
                                self.session.global_log.loc[idx, 'correct'] = 1
                                self.parameters['correct'] = 1

                            elif self.parameters['signal_present'] !=  self.session.global_log.loc[idx, 'response']:
                                self.session.global_log.loc[idx, 'correct'] = 0
                                self.parameters['correct'] = 0

                            self.stop_phase()

                    elif ev == self.session.response_button_mapping['absent_not_confident']:

                        if self.phase == 3:
                            idx = self.session.global_log.shape[0]
                            stim_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.ID) & \
                                                                    (self.session.global_log['event_type']=='stimulus_window'), 'onset'].to_numpy()[0]

                            RT = t - stim_onset

                            self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                            self.session.global_log.loc[idx, 'onset'] = t
                            self.session.global_log.loc[idx, 'event_type'] = 'response'
                            self.session.global_log.loc[idx, 'phase'] = self.phase
                            self.session.global_log.loc[idx, 'key'] = ev
                            self.session.global_log.loc[idx, 'nr_frames'] = self.session.nr_frames 
                            self.session.global_log.loc[idx, 'response'] = 0
                            self.session.global_log.loc[idx, 'confidence'] = 0
                            self.session.global_log.loc[idx, 'RT'] = RT

                            self.parameters['response'] = 0
                            self.parameters['confidence'] = 0
                            self.parameters['RT'] = RT

                            if self.parameters['signal_present'] ==  self.session.global_log.loc[idx, 'response']:
                                self.session.global_log.loc[idx, 'correct'] = 1
                                self.parameters['correct'] = 1

                            elif self.parameters['signal_present'] !=  self.session.global_log.loc[idx, 'response']:
                                self.session.global_log.loc[idx, 'correct'] = 0
                                self.parameters['correct'] = 0 

                            self.stop_phase()

                    elif ev == self.session.response_button_mapping['present_not_confident']:

                        if self.phase == 3:
                            idx = self.session.global_log.shape[0]
                            stim_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.ID) & \
                                                                    (self.session.global_log['event_type']=='stimulus_window'), 'onset'].to_numpy()[0]

                            RT = t - stim_onset

                            self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                            self.session.global_log.loc[idx, 'onset'] = t
                            self.session.global_log.loc[idx, 'event_type'] = 'response'
                            self.session.global_log.loc[idx, 'phase'] = self.phase
                            self.session.global_log.loc[idx, 'key'] = ev
                            self.session.global_log.loc[idx, 'nr_frames'] = self.session.nr_frames 
                            self.session.global_log.loc[idx, 'response'] = 1
                            self.session.global_log.loc[idx, 'confidence'] = 0
                            self.session.global_log.loc[idx, 'RT'] = RT

                            self.parameters['response'] = 1
                            self.parameters['confidence'] = 0
                            self.parameters['RT'] = RT

                            if self.parameters['signal_present'] ==  self.session.global_log.loc[idx, 'response']:
                                self.session.global_log.loc[idx, 'correct'] = 1
                                self.parameters['correct'] = 1

                            elif self.parameters['signal_present'] !=  self.session.global_log.loc[idx, 'response']:
                                self.session.global_log.loc[idx, 'correct'] = 0
                                self.parameters['correct'] = 0

                            self.stop_phase()

                    elif ev == self.session.response_button_mapping['present_confident']:

                        if self.phase == 3:
                            idx = self.session.global_log.shape[0]
                            stim_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.ID) & \
                                                                    (self.session.global_log['event_type']=='stimulus_window'), 'onset'].to_numpy()[0]
                            
                            RT = t - stim_onset

                            self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                            self.session.global_log.loc[idx, 'onset'] = t
                            self.session.global_log.loc[idx, 'event_type'] = 'response'
                            self.session.global_log.loc[idx, 'phase'] = self.phase
                            self.session.global_log.loc[idx, 'key'] = ev
                            self.session.global_log.loc[idx, 'nr_frames'] = self.session.nr_frames 
                            self.session.global_log.loc[idx, 'response'] = 1
                            self.session.global_log.loc[idx, 'confidence'] = 1                
                            self.session.global_log.loc[idx, 'RT'] = RT

                            self.parameters['response'] = 1
                            self.parameters['confidence'] = 1
                            self.parameters['RT'] = RT

                            if self.parameters['signal_present'] ==  self.session.global_log.loc[idx, 'response']:

                                self.session.global_log.loc[idx, 'correct'] = 1
                                self.parameters['correct'] = 1

                            elif self.parameters['signal_present'] !=  self.session.global_log.loc[idx, 'response']:

                                self.session.global_log.loc[idx, 'correct'] = 0
                                self.parameters['correct'] = 0

                            self.stop_phase()
                
                elif self.session.settings['response']['metacognition'] == 'split':
                
                    if ev in ['esc', 'escape', 'q']:
                        self.stopped = True
                        self.session.stopped = True

                        print('run canceled by user')
                        self.stop_trial()
                        if self.phase == 0:
                            self.first_trial_hold = False
                            self.stop_phase()

                    elif (ev == 'space') or (ev == 'return'):
                        if self.phase == 0:
                            self.first_trial_hold = False
                            self.stop_phase()
                    


                    # present/absent response
                    elif ev == self.session.response_button_mapping['absent']:
                        if self.phase == 3: # present/absent response

                            idx = self.session.global_log.shape[0]
                            stim_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.ID) & \
                                                                    (self.session.global_log['event_type']=='stimulus_window'), 'onset'].to_numpy()[0]

                            RT = t - stim_onset

                            self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                            self.session.global_log.loc[idx, 'onset'] = t
                            self.session.global_log.loc[idx, 'event_type'] = 'response'
                            self.session.global_log.loc[idx, 'phase'] = self.phase
                            self.session.global_log.loc[idx, 'key'] = ev
                            self.session.global_log.loc[idx, 'nr_frames'] = self.session.nr_frames 
                            self.session.global_log.loc[idx, 'response'] = 0
                            self.session.global_log.loc[idx, 'RT'] = RT

                            self.parameters['response'] = 0
                            self.parameters['RT'] = RT

                            if self.parameters['signal_present'] ==  self.session.global_log.loc[idx, 'response']:
                                self.session.global_log.loc[idx, 'correct'] = 1
                                self.parameters['correct'] = 1

                            elif self.parameters['signal_present'] !=  self.session.global_log.loc[idx, 'response']:
                                self.session.global_log.loc[idx, 'correct'] = 0
                                self.parameters['correct'] = 0
                            self.stop_phase()
                            
                    elif ev == self.session.response_button_mapping['present']:
                        if self.phase == 3: 
                    
                            idx = self.session.global_log.shape[0]
                            stim_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.ID) & \
                                                                    (self.session.global_log['event_type']=='stimulus_window'), 'onset'].to_numpy()[0]

                            RT = t - stim_onset

                            self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                            self.session.global_log.loc[idx, 'onset'] = t
                            self.session.global_log.loc[idx, 'event_type'] = 'response'
                            self.session.global_log.loc[idx, 'phase'] = self.phase
                            self.session.global_log.loc[idx, 'key'] = ev
                            self.session.global_log.loc[idx, 'nr_frames'] = self.session.nr_frames 
                            self.session.global_log.loc[idx, 'response'] = 1
                            self.session.global_log.loc[idx, 'RT'] = RT

                            self.parameters['response'] = 1
                            self.parameters['RT'] = RT

                            if self.parameters['signal_present'] ==  self.session.global_log.loc[idx, 'response']:
                                self.session.global_log.loc[idx, 'correct'] = 1
                                self.parameters['correct'] = 1

                            elif self.parameters['signal_present'] !=  self.session.global_log.loc[idx, 'response']:
                                self.session.global_log.loc[idx, 'correct'] = 0
                                self.parameters['correct'] = 0
                            self.stop_phase()

                    
                    # metacog response
                    elif ev == self.session.response_button_mapping['low_confidence']:
                        if self.phase == 5: 

                            idx = self.session.global_log.shape[0]
                            stim_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.ID) & \
                                                                    (self.session.global_log['event_type']=='response_window_metacog'), 'onset'].to_numpy()[0]

                            RT = t - stim_onset

                            self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                            self.session.global_log.loc[idx, 'onset'] = t
                            self.session.global_log.loc[idx, 'event_type'] = 'confidence_response'
                            self.session.global_log.loc[idx, 'phase'] = self.phase
                            self.session.global_log.loc[idx, 'key'] = ev
                            self.session.global_log.loc[idx, 'nr_frames'] = self.session.nr_frames 
                            self.session.global_log.loc[idx, 'confidence'] = 0
                            self.session.global_log.loc[idx, 'RT'] = RT

                            self.parameters['confidence'] = 0
                            self.parameters['RT'] = RT
                            self.stop_phase()



                    elif ev == self.session.response_button_mapping['high_confidence']:
                        if self.phase == 5: 
                    

                            idx = self.session.global_log.shape[0]
                            stim_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.ID) & \
                                                                    (self.session.global_log['event_type']=='response_window_metacog'), 'onset'].to_numpy()[0]

                            RT = t - stim_onset

                            self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                            self.session.global_log.loc[idx, 'onset'] = t
                            self.session.global_log.loc[idx, 'event_type'] = 'confidence_response'
                            self.session.global_log.loc[idx, 'phase'] = self.phase
                            self.session.global_log.loc[idx, 'key'] = ev
                            self.session.global_log.loc[idx, 'nr_frames'] = self.session.nr_frames 
                            self.session.global_log.loc[idx, 'confidence'] = 1
                            self.session.global_log.loc[idx, 'RT'] = RT

                            self.parameters['confidence'] = 1
                            self.parameters['RT'] = RT
                            self.stop_phase()




    def pupil_check(self):

        # Track if participants blinked during critical phases of the trial (baseline, stimulus, response)
        pupil = self.session.tracker.getNewestSample().getLeftEye().getPupilSize()

        if self.phase == 1:
            if pupil == 0:  # if blink, mark the trial as bad 
                self.blinkDuringTrial = True
                self.parameters['blinkDuringTrial'] = 1

        elif self.phase == 2: 
            if pupil == 0: # if blink, mark the trial as bad 
                self.blinkDuringTrial = True
                self.parameters['blinkDuringTrial'] = 1

        elif self.phase == 3:
            if pupil == 0: # if blink, mark the trial as bad 
                self.blinkDuringTrial = True
                self.parameters['blinkDuringTrial'] = 1

    def stimulus_check(self):
        """
        Draws all stimuli next to each other.
        Only to be used in isolation, as it messess with stimulus positions
        """

        for i, (sig, stim) in enumerate(self.session.stimuli.items()):

            if self.stim_set == False:

                if i%5 == 0:
                    x = -500
                elif i%5 == 1:
                    x = -250
                elif i%5 == 2:
                    x = 0
                elif i%5 == 3:
                    x = 250
                elif i%5 == 4:
                    x = 500

                if i<5 :
                    y = 250
                else:
                    y = -250

                stim.pos = (x, y)
        
        self.stim_set = True

        for i, (sig, stim) in enumerate(self.session.stimuli.items()):
                                
            stim.draw()

    def run(self):

        """ Runs through phases. Should not be subclassed unless
        really necessary. """

        if self.eyetracker_on:  # Sets status message
            cmd = f"record_status_message 'trial {self.trial_nr}'"
            self.session.tracker.sendCommand(cmd)

        # Because the first flip happens when the experiment starts,
        # we need to compensate for this during the first trial/phase
        if self.session.first_trial:
            # must be first trial/phase
            if self.timing == 'seconds':  # subtract duration of one frame
                self.phase_durations[0] -= 1./self.session.actual_framerate * 1.1  # +10% to be sure

            else:  # if timing == 'frames', subtract one frame 
                self.phase_durations[0] -= 1

            self.session.first_trial = False

        for phase_dur in self.phase_durations:  # loop over phase durations
            # pass self.phase *now* instead of while logging the phase info.
            self.session.win.callOnFlip(self.log_phase_info, phase=self.phase)

            # Start loading in next trial during this phase (if not None)
            if self.load_next_during_phase == self.phase:
                self.load_next_trial(phase_dur)

            if self.timing == 'seconds':
                # Loop until timer is at 0!
                self.session.timer.addTime(-phase_dur)
                self.first_trial_hold = True
                if (self.ID==0 or self.ID % self.session.block_len == 0) and self.phase==0 :
                    while self.first_trial_hold:

                        self.draw()
                        if self.draw_each_frame:
                            self.session.win.flip()
                            self.session.nr_frames += 1
                            self.session.nr_frames_no_reset += 1

                        # if self.eyetracker_on:
                        #     self.pupil_check()

                        self.event()

                while self.session.timer.getTime() < 0 and not self.exit_phase and not self.exit_trial:
                    self.draw()
                    if self.draw_each_frame:
                        self.session.win.flip()
                        self.session.nr_frames += 1
                        self.session.nr_frames_no_reset += 1

                    # if self.eyetracker_on:
                    #     self.pupil_check()

                    self.event()

            if self.exit_phase:  # broke out of phase loop
                self.session.timer.reset()  # reset timer!
                self.exit_phase = False  # reset exit_phase

            if self.exit_trial:
                self.session.timer.reset()
                break

            self.phase += 1  # advance phase


class DetectSession(PylinkEyetrackerSession):

    def __init__(self, output_str, output_dir, settings_file=None, eyetracker_on=True):
        super().__init__(output_str, output_dir, settings_file, eyetracker_on=eyetracker_on) # initialize parent class

        subnr = output_str.split('_')[0].split('-')[1]

        self.subnr = subnr
        self.nr_frames_no_reset = 0
        # from future exptools2 version, Session init
        self.width_deg = 2 * np.degrees(
            np.arctan(self.monitor.getWidth() / self.monitor.getDistance())
        )
        self.pix_per_deg = self.win.size[0] / self.width_deg
        self.settings_file = settings_file
        # self.noise_sizePIX = np.round(noise_parameters['size'] * self.pix_per_deg / 2).astype(int) # <= notice division by 2, it's to increase the noise element size to 2 later

        self.signal_parameters = self.settings['stimuli']['signal_parameters']
        self.noise_parameters = self.settings['stimuli']['noise_parameters']  

        # TODO use this in make_trials to make balanced (mini) blocks, clean up deprecated code pertaining to old way 
        self.signals = np.array(self.signal_parameters['signals']) 
        self.n_signals = len(self.signals)
        self.n_repeats_block = self.settings['task']['n_repeats_block']
        self.ratio_absent_present = self.settings['task']['ratio_absent_present']
        self.n_blocks = self.settings['task']['n_blocks']
        self.block_len = (self.n_signals * self.n_repeats_block) + int(self.n_signals * self.n_repeats_block * self.ratio_absent_present)
        self.max_trials = self.block_len * self.n_blocks
        
        print(f"{self.n_signals} signals repeated {self.n_repeats_block} times per block with {int(self.n_signals * self.n_repeats_block * self.ratio_absent_present)}\
              absent trials makes a block length of {self.block_len}. {self.n_blocks} blocks specified for a total {self.n_blocks * self.block_len} trials.")
        
        dot_size = self.settings['stimuli']['dot_size']

        self.green_fix = Circle(self.win, radius=dot_size, edges = 100, lineWidth=0)
        self.green_fix.setColor((0, 128, 0), 'rgb255')
        self.black_fix = Circle(self.win, radius=dot_size, edges = 100, lineWidth=0)
        self.black_fix.setColor((0, 0, 0), 'rgb255')
        self.white_fix = Circle(self.win, radius=dot_size, edges = 100, lineWidth=0)
        self.white_fix.setColor((255, 255, 255), 'rgb255')
        self.blue_fix = Circle(self.win, radius=dot_size, edges = 100, lineWidth=0)
        self.blue_fix.setColor((0, 171, 240), 'rgb255')
        
        ## for screenshots, make fix bigger
        if self.settings['stimuli']['screenshot']:
            self.green_fix.setSize(5)
            self.black_fix.setSize(5)
            self.white_fix.setSize(5)

        # make response button mapping
        if self.settings['response']['device'] == 'keyboard':
            if self.settings['response']['metacognition'] == 'together': 
                self.response_button_mapping = {'present_confident' : 'l',
                                                'present_not_confident' : 'k',
                                                'absent_confident' : 'a',
                                                'absent_not_confident' : 's'}

            elif self.settings['response']['metacognition'] == 'split':
                self.response_button_mapping = {'present' : 'k',
                                                'absent' : 'a',
                                                'high_confidence' : 'm',
                                                'low_confidence' : 'z'
                                                }
            else:
                raise ValueError(f"{self.settings['response']['metacognition']} is not a valid setting for ['response']['metacognition']")

        elif self.settings['response']['device'] == 'button_box':
            if self.settings['response']['metacognition'] == 'together': 
                self.response_button_mapping = {'present_confident' : '2',
                                            'present_not_confident' : '4',
                                            'absent_confident' : '1',
                                            'absent_not_confident' : '3'}

            elif self.settings['response']['metacognition'] == 'split':
                self.response_button_mapping = {'present' : '2',
                                                'absent' : '4',
                                                'high_confidence' : '1',
                                                'low_confidence' : '3'
                                                }
            else:
                raise ValueError(f"{self.settings['response']['metacognition']} is not a valid setting for ['response']['metacognition']")

        # load noise textures into noise stimuli
        self.n_noise_textures = self.settings['stimuli']['n_noise_textures']
        self.noise_stimuli = []

        for i in range(self.n_noise_textures):

            path = f'textures/binary_noise/binary_noise_id-{str(i).zfill(4)}.bmp'
            stim = ImageStim(self.win, image = path,
                            contrast= self.noise_parameters['contrast'], name='noise', units='pix', 
                            mask='raisedCos', #size=(self.noise_sizePIX*2, self.noise_sizePIX*2), 
                            opacity= self.noise_parameters['opacity'],)
            
            self.noise_stimuli.append(stim)

            # print(f'made texture from id {i} with width {stim.width} and height {stim.height}')

        # make trials
        self.create_yes_no_trials()


    def create_yes_no_trials(self):
        """creates trials for yes/no runs"""

        ##############
        # TODO use this in make_trials to make balanced (mini) blocks, clean up deprecated code pertaining to old way 
        # self.signals = np.array(self.signal_parameters['signals']) 
        # self.n_signals = len(self.signals)
        # self.n_repeats_block = self.settings['task']['n_repeats_block']
        # self.ratio_absent_present = self.settings['task']['ratio_absent_present']
        # self.n_blocks = self.settings['task']['n_blocks']
        # self.block_len = (self.n_signals * self.n_repeats_block) + (self.n_signals * self.n_repeats_block * self.ratio_absent_present)
        # self.max_trials = self.block_len * self.n_blocks
        
        # print(f"{self.n_signals} signals repeated {self.n_repeats_block} times per block with {self.n_signals * self.n_repeats_block * self.ratio_absent_present}\
        #       absent trials makes a block length of {self.block_len}. {self.n_blocks} blocks specified for a total {self.n_blocks * self.block_len} trials.")        
        # #############
        
        # init standard parameters
        self.standard_parameters = {'subject': self.subnr}

        # Amount of signal present trials
        present_trials = self.max_trials/2
        # total amount of signal repeats
        signal_repetitions = present_trials/self.n_signals

        if not signal_repetitions == int(signal_repetitions):
            raise ValueError('Signal strengths not balanced')

        signal_repetitions = int(signal_repetitions)

        # print('Average signal strength: {:.3f}'.format(self.signals.mean(), 3))
        # print('Unique signals: {}'.format(self.signals))
        
        strong_opacity = .1
        # stimuli will be held here
        self.stimuli = {}
        self.mask = self.makeRingGaussian(self.signal_parameters['size'],  sd=.35, r=150)[0]
        # print(f"created mask with size {self.mask.shape} and values in range {self.mask.min(), self.mask.max()}")
        
        for signal_opacity in self.signals:
            # making all stimuli, put them into self.simuli
            tex = self.make_grating(1024, #self.stim_sizePIX
                                    ori = 60, sf = self.signal_parameters['spatial_freq'], unit = 'deg')
            # print(f"created tex with size {tex.shape} and values in range {tex.min(), tex.max()}")
            # print(f"tex * mask with size {(tex * self.mask).shape} and values in range {(tex * self.mask).min(), (tex * self.mask).max()}")
        
            grating = GratingStim(self.win, contrast = self.signal_parameters['contrast'], opacity= signal_opacity,
                                        tex = tex * self.mask, mask='raisedCos', units='pix', size=self.signal_parameters['size'],#self.stim_sizePIX,# sf = self.signal_parameters['spatial_freq'],
                                        color=[1,1,1]) 
            self.stimuli[signal_opacity] = grating

        # add 0-stimulus
        self.stimuli[0] = GratingStim(self.win, contrast = self.signal_parameters['contrast'], opacity=0,
                                        tex = tex * self.mask, mask='raisedCos', units='pix', size=self.signal_parameters['size'],#self.stim_sizePIX,# sf = self.signal_parameters['spatial_freq'],
                                        color=[1,1,1])
        # add strong stimulus for tutorial trials
        self.stimuli[strong_opacity] = GratingStim(self.win, contrast = self.signal_parameters['contrast'], opacity=strong_opacity,
                                        tex = tex * self.mask, mask='raisedCos', units='pix', size=self.signal_parameters['size'],#self.stim_sizePIX,# sf = self.signal_parameters['spatial_freq'],
                                        color=[1,1,1])

        self.miniblock_params = []
        # make empty trials
        for empty_trial in range(int(self.n_signals * self.ratio_absent_present)):
            # make empty trial
            # copy params from template
            params = self.standard_parameters.copy()
            # update whether signal present or not, signal ori 
            params.update({'signal_present':0, 'signal_orientation':0, 'signal_opacity' : 0})

            self.miniblock_params.append(params.copy())

        for signal in self.signals:
            # make signal trial 
            # copy params from template
            params = self.standard_parameters.copy()
            # update whether signal present or not, signal ori 
            params.update({'signal_present':1, 'signal_orientation':0, 'signal_opacity' : signal})

            # self.miniblock_params_durs.append([params.copy(), np.array(phase_durs)])
            self.miniblock_params.append(params.copy())

        self.trial_parameters = []
        for i in range(self.n_repeats_block * self.n_blocks):
            random.shuffle(self.miniblock_params)
            self.trial_parameters += self.miniblock_params
        
        self.total_duration = 0
        # constructing phase durations
        self.durs = []
        for i in range(len(self.trial_parameters)):
            if self.settings['response']['metacognition'] == 'together': 
            
                # phase durations, and iti's:
                # phases: 0=pretrial, 1=baseline, 2=stim, 3=response, 4=feedback 5=ITI
                phase_durs = [-0.01, 0.5, 0.2, 2, 0.3, np.random.uniform(1, 1.4)]
                self.durs.append(phase_durs) 
            
            elif self.settings['response']['metacognition'] == 'split':
                # phase durations, and iti's:
                # phases: 0=pretrial, 1=baseline, 2=stim, 3=response, 4=feedback, 5=resp_meta 6=feedback_meta 7=ITI
                # phase_durs = [-0.01, 0.5, 0.2, 1.5, 0.3, 1.5, 0.3, np.random.uniform(0.6, 1.0)]
                phase_durs = [-0.01, 0.5, 0.2, 2, 0.3, 2, 0.3, np.random.uniform(1, 1.4)]
                self.durs.append(phase_durs)
            
            else:
                raise ValueError(f"{self.settings['response']['metacognition']} is not a valid setting for ['response']['metacognition']")
            
            self.total_duration += np.array(phase_durs).sum()

        self.trial_parameters_and_durs = list(zip(self.trial_parameters, self.durs))
        # print("total duration: %.2f min." % (self.total_duration / 60.0))

        # contructing phase names
        if self.settings['response']['metacognition'] == 'together': 
            self.phase_names = ['blockStart','baseline_window','stimulus_window','response_window','feedback_window', 'ITI_window']
        
        elif self.settings['response']['metacognition'] == 'split': 
            self.phase_names = ['blockStart','baseline_window','stimulus_window','response_window','feedback_window',
                                'response_window_metacog','feedback_window_metacog', 'ITI_window']
        else:
            raise ValueError(f"{self.settings['response']['metacognition']} is not a valid setting for ['response']['metacognition']")

        self.corrects = []
        self.confidence = []

        # Get index of first trials of each block
        # print(self.max_trials)
        ftib = np.arange(0, self.max_trials, self.block_len)

        # Now, loop over all trials (in random order) to initialize the Trial classes. Later we can then just run the pre-initialised 
        # Trial objects, so that the transition from trial N to trial N+1 goes more swiftly. This should minimize the delay in noise presentation.

        self.Trials2Run = []
        # signals = np.sort(signals)
        for i, params_durs in enumerate(self.trial_parameters_and_durs):

            if np.isin(i, ftib): # For first trial in block, add some time before target stimulus onset
                params_durs[1][1] += 1

            self.Trials2Run.append(DetectTrial(parameters=params_durs[0], phase_durations=params_durs[1],\
                                      session=self,monitor=self.monitor,  ID=i, phase_names = self.phase_names))

        # make tutorial trials
        # TODO
        if self.settings['task']['tutorial']:

            ## hardcoded tutorial timings
            phase_durs_tut = [[180, 0, 0, 0, 0, 0, 0, 0], # tutorial screen
                              [[-0.01, 60, 60, 60, 0.3, 60, 0.3, np.random.uniform(1, 1.4)] for _ in range(6)][0], # self paced
                              [[-0.01, 60, 60, 60, 0.3, 60, 0.3, np.random.uniform(1, 1.4)] for _ in range(6)][0], # real signals
                              [[-0.01, 0.5, 0.2, 2, 0.3, 2, 0.3, np.random.uniform(1, 1.4)] for _ in range(6)][0], # real timing
                              ]
            
            signal_present_tut = [0, # tutorial screen
                              1, 0, 1, 0, 1, 0, # self paced
                              1, 0, 1, 0, 1, 0, # real signals
                              1, 0, 1, 0, 1, 0, # real timing
                              ]
            # signals = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
            
            signal_opacity_tut = [0, # tutorial screen
                              .05, 0, .04, 0, .03, 0, # self paced
                              .045, 0, .035, 0, .025, 0, # real signals
                              .03, 0, .045, 0, .015, 0, # real timing
                              ]
            
            print(phase_durs_tut)


            unique_signals = np.array(self.signal_parameters['signals']) 
            self.tutorial_trials = []
            for i, phase_durs in enumerate(phase_durs_tut):
                print(phase_durs)
                params = self.standard_parameters.copy()

                if 1 <= i < 6:
                    params.update({'signal_present':1,
                                'signal_orientation': 0,
                                'signal_opacity':unique_signals[-1],
                                'tutorial_text':'a strong signal'})
                elif i % 3 == 1:
                    params.update({'signal_present':1,
                                'signal_orientation': 0,
                                'signal_opacity':unique_signals[3],
                                'tutorial_text':'a weak signal'})
                else:
                    params.update({'signal_present':0,
                                'signal_orientation': 0,
                                'signal_opacity':0,
                                'tutorial_text':'signal absent'})

                # print(f"trial pre -{i}, signal {unique_signals[i]}")
                    # print(self.trial_parameters_and_durs[i][0])
                #    print(self.trial_parameters_and_durs)
                
                # phases: 0=pretrial, 1=baseline, 2=stim, 3=response, 4=feedback, 5=resp_meta 6=feedback_meta 7=ITI
                
                # phase_durs = phase_durs_tut[0]

                self.tutorial_trials.append(DetectTrial(parameters=params, phase_durations=phase_durs,
                                        session=self,monitor=self.monitor,  ID=-i, phase_names = self.phase_names))

            self.Trials2Run = self.tutorial_trials + self.Trials2Run        
        print(f"Made {len(self.Trials2Run)} trials to run")
        print("total duration: %.2f min." % (self.total_duration / 60.0))

    def run(self):
        """run the session"""
        # Cycle through trials
        self.start_experiment()
        self.stopped = False
        self.session_parameters = []

        if self.eyetracker_on:
            self.calibrate_eyetracker()
            self.start_recording_eyetracker()

        fixlostcount = 0

        # loop over trials
        for t in range(self.max_trials):
            self.stopped = False
            print(t)
            # Run trial
            # shell()
            self.Trials2Run[t].run()    

            # Print message to terminal, if participant blinked
            if self.Trials2Run[t].parameters['blinkDuringTrial']:
                fixlostcount +=1
                print(str(fixlostcount) + " trials with lost fixation")

            self.corrects.append(self.Trials2Run[t].parameters['correct'])
            self.confidence.append(self.Trials2Run[t].parameters['confidence'])
            self.session_parameters.append(self.Trials2Run[t].parameters)

            if self.stopped == True:
                break

        self.close()

    def makeRingGaussian(self, size, sd = 3, r=10, center=None):
        """ Make a square gaussian kernel.

        size is the length of a side of the square in PX
        sd is standard deviation
        r is the radius of the ring around which the Gaussian is computed
        
        adapted from: https://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussian-with-python
        """

        x = np.arange(0, size, 1, float)
        y = x[:,np.newaxis]

        if center is None:
            x0 = y0 = size // 2
        else:
            x0 = center[0]
            y0 = center[1]
        distance = np.abs(np.sqrt((x-x0)**2 + (y-y0)**2) - r)
        normalizedDist = (distance-np.min(distance))/(np.max(distance)-np.min(distance)) 
        ring_mask = 1/(2*np.pi *sd**2) * np.exp(-normalizedDist / (2*sd**2))
        ring_mask_scaled = (ring_mask - ring_mask.min())/(ring_mask.max()-ring_mask.min())

        return ring_mask_scaled, normalizedDist
    
    def make_grating(self, size, ori = 90, sf = 100, phase = 0, unit = 'pixels', dva_in_px=94.1):
        """
        adapted from: https://www.baskrahmer.nl/blog/generating-gratings-in-python-using-numpy
        
        :param sf: spatial frequency (in pixels) 
        :param ori: wave orientation (in degrees, [0-360])
        :param phase: wave phase (in degrees, [0-360])
        :param size: image size (integer)
        :param unit: unit of sf: ['pix', 'pixels', 'dva', 'deg']
        :param dva_in_px: how many pix are one dva. default is in psychophysics room
        :return: numpy array of shape (imsize, imsize)
        
        """
        assert unit in ['pix', 'pixels', 'dva', 'deg'], f"{unit} is not a valid unit"

        # converting dva to pixels
        if (unit == 'dva') or (unit == 'deg'):
            sf = dva_in_px / sf

        x, y = np.meshgrid(np.arange(size), np.arange(size))
        # Get the appropriate gradient
        gradient = np.sin(ori * np.pi / 180) * x - np.cos(ori * np.pi / 180) * y
        # Plug gradient into wave function
        grating = np.sin((2 * np.pi * gradient) / sf + (phase * np.pi) / 180)
        
        return grating


    def close(self):
        
        if self.settings['stimuli']['screenshot']:
            print('saving movie')
            self.win.saveMovieFrames('movie.tif')

        super().close()


    def post_process(self, plot = True):
        ## post-processing
        # load data
        data_path = op.join(self.output_dir, self.output_str + "_events.tsv")
        df = pd.read_csv(data_path, sep='\t')
        # process
        ops = []
        resps = []
        for trial in df.trial_nr.unique():
        #     print(trial)
            trial_df = df[df.trial_nr == trial]
            opacity = trial_df.signal_opacity.iloc[0]
            try:
                response_correct = trial_df[trial_df.event_type == 'response'].correct.values[0]
            except IndexError:
                response_correct = None
            ops.append(opacity)
            resps.append(response_correct)
        
        # fit and plot
        _, fig, ax = quick_fit(ops, resps, gaussian, init = {'$\mu$':0.02,'$\sigma$':.005},
                   plot = True)

        # title
        fig.suptitle(self.output_str)
        # show
        plt.show()
        # save
        plt.savefig(op.join(self.output_dir, self.output_str + "_quickfit.png"), fig)

        

def main(block=0):

    subject = sys.argv[1]
    sess =  sys.argv[2]
    task = 'SigDet' # different settings -> now implemented as saving the actual settings
    run = sys.argv[3] # which run    
    output_str = f'sub-{subject}_sess-{sess}_task-{task}_run-{run}'

    # save results
    results_folder = f'{task}_pilot/sub-{subject}/ses-{sess}'
    # print(results_folder) 

    # Check if the directory already exists
    if not os.path.exists(results_folder):
        # Create the directory
        os.makedirs(results_folder)
        print("results_folder created successfully!")
    else:
        print("results_folder already exists!")
    
    # output folder + string is handled as one by the task 
    output_str = os.path.join(results_folder, output_str)

    ts = DetectSession(output_str=output_str, subnr=subject, index=sess, 
                       block=block, eyetracker_on=False)
    # print(ts.settings)
    ts.run()

    # and save
    try:
        pd.DataFrame(ts.session_parameters).to_csv(os.path.join(results_folder, f'sub-{subject}_sess-{sess}_task-{task}_run-{run}_parameters.csv'))
    except AttributeError:
        print("Caught attribute error, when saving 'session_parameters'. Maybe session did not run?")
    
if __name__ == '__main__':

    main(block = 0)
