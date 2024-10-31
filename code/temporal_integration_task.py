# switching to PTB
from psychopy import prefs
prefs.hardware['audioLib'] ='PTB'
prefs.hardware['audioLatencyMode'] = 3
# prefs.hardware['audioDevice'] ='HDA Intel PCH: ALC262 Analog (hw:0,0)'

# from psychopy import sound

from exptools2.core import PylinkEyetrackerSession
from exptools2.core import Trial
from exptools2.stimuli import create_circle_fixation

from psychopy.visual import TextStim, ImageStim, Circle
from psychopy import event
from psychopy import visual, core
from psychopy import sound
# from psychopy.sound import Microphone

import psychtoolbox as ptb

import random
import pyglet
import pandas as pd
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools
import glob
from psychopy.visual import Rect
from .utils import quick_fit, gaussian, cumulative_normal_lapse

class TempIntTrial(Trial):
    """ Simple trial with text (trial x) and fixation. """
    def __init__(self, session, trial_nr, phase_durations, txt=None, parameters=None, **kwargs):
        super().__init__(session, trial_nr, phase_durations, **kwargs)
        self.txt = TextStim(self.session.win, txt) 

        self.img = ImageStim(self.session.win, self.session.texture_paths[trial_nr%len(self.session.texture_paths)], size = 10,
                             mask = 'raisedCos', maskParams = {'fringeWidth':0.2}) # proportion that will be blurred
        
        self.sound_played = False
        self.parameters = parameters
        self.parameters['response_given'] = False
        self.parameters['confidence_response_given'] = False

        self.draw_visual = False if self.parameters['order'] == 'AV' else True
        self.frames_drawn_visual = 0 # counter to draw visual stimulus

        # set frame index of start second stimulus: duration first + soa - 1 for proper indexing
        self.start_second_stimulus_frame = self.session.stim_dur_aud + self.parameters['soa'] - 1

        # set frame index of start second stimulus: (soa-duration first) - 1 for proper indexing
        self.start_second_stimulus_frame = self.parameters['soa'] #- 1

        if trial_nr == 0:
            # set intro message
            if self.session.settings['task']['type'] == 'TOJ':
                self.intro = ImageStim(self.session.win, 'assets/intructions_screen_TempIntTOJ_seperate.png', units='pix', size = [int(1920 * .95), int(1080*.95)])
            
            else:
                self.intro = ImageStim(self.session.win, 'assets/intructions_screen_TempIntSJ_seperate.png', units='pix', size = [int(1920 * .95), int(1080*.95)])

        elif self.trial_nr % self.session.n_trials_block==0:
            pause_text = f"Great, you did {self.trial_nr} of {self.session.n_trials} trials.\nYou can rest a little now, but try to keep your head stable.\nContinue with any button if you're ready."
            self.pause_message = visual.TextStim(self.session.win, pos=[0,0], text= pause_text, color = (1.0, 1.0, 1.0), height=0.5, font='Arial', wrapWidth=850)

                # squares for Photodiode
        if self.session.photodiode_check is True:
            self.white_square = Rect(self.session.win, 2, 2, pos = (5.5,2.5))
            self.black_square = Rect(self.session.win, 2, 2, pos = (5.5,2.5), fillColor = 'black')
            # if self.parameters['trial_type'] == 'dur':
            #     self.square_flip_frames = self.session.var_dur_dict_flip[self.parameters['stim_dur']]
            #     print(self.square_flip_frames)
            # else:
            #     self.square_flip_frames = self.session.var_isi_dict_flip[self.parameters['stim_dur']]

        if self.session.photodiode_check:
            self.recorded = False

        print(f'initialized trial {trial_nr} with durations {phase_durations} \n\
              and second stim start {self.start_second_stimulus_frame} and parameters {parameters}')

    def draw(self):
        """ Draws stimuli 
        potentially, the sound_played flag can be removed now as we are only playing on one specific frame
        """

        # debug message
        if self.session.debug:
            self.session.debug_message.setText(f"trial {self.trial_nr}, phase {self.phase}\norder {self.parameters['order']}, soa {self.parameters['soa']}")
            self.session.debug_message.draw()

        if (self.trial_nr == 1) and self.session.settings['stimuli']['screenshot']:
            print('getting Movie frame')
            self.session.win.getMovieFrame()

        if self.phase == 0:
            if self.trial_nr == 0:
                # first trial, intro screen
                self.intro.draw()
            elif self.trial_nr % self.session.n_trials_block==0:
                # block pause
                self.pause_message.draw()

        if self.phase == 1:
            # jittered blank
            self.session.default_fix.draw()
            
            if self.session.photodiode_check:
                self.black_square.draw()

        elif self.phase == 2:
            if self.session.photodiode_check:
                self.black_square.draw()
            
            if self.parameters["order"] == "VA":
                # visual first
                if self.draw_visual:
                    self.img.draw()
                    # if self.session.photodiode_check & ((self.frames_drawn_visual == 0) or (self.frames_drawn_visual == 3)):
                    # if (self.frames_drawn_visual == 0):# or (self.frames_drawn_visual == 3):
                    if self.session.photodiode_check:
                        self.white_square.draw()
                    # else:
                        # self.black_square.draw()

                    self.frames_drawn_visual += 1
                    
                    # print(f"on frame {self.session.on_phase_frame}, drew visual {self.frames_drawn_visual}")
                    if self.frames_drawn_visual == self.session.stim_dur_vis:
                        self.draw_visual = False
                
                # audio second
                if self.start_second_stimulus_frame == self.session.on_phase_frame:
                    if not self.sound_played:
                        # print('play sound')
                        # now = ptb.GetSecs()
                        # self.mySound.play(when=now+4)  # play in EXACTLY 4s

                        # schedule sound
                        nextFlip = self.session.win.getFutureFlipTime(clock='ptb')

                        self.session.sound.play(when=nextFlip)

                        # self.session.sound.play()

                        # print(f"on frame {self.session.on_phase_frame}, playing sound at {nextFlip}")

                        self.sound_played = True


            elif self.parameters["order"] == "AV":
                # audio first
                if not self.sound_played:
                    # print('play sound')
                    # now = ptb.GetSecs()
                    # self.mySound.play(when=now+4)  # play in EXACTLY 4s
                    nextFlip = self.session.win.getFutureFlipTime(clock='ptb')
                    
                    self.session.sound.play(when=nextFlip)

                    # self.session.sound.play()
                    self.sound_played = True
                
                # visual second
                if self.start_second_stimulus_frame == self.session.on_phase_frame:
                    self.draw_visual = True
                
                if self.draw_visual:
                    self.img.draw()

                    if self.session.photodiode_check:
                        self.white_square.draw()

                    self.frames_drawn_visual += 1

                    if self.frames_drawn_visual == self.session.stim_dur_vis:
                        self.draw_visual = False

            else:
                raise ValueError(f"The only supported stimulus orders are 'VA' and 'AV'. You requested {self.parameters['order']}")
        
            # always draw default fix
            self.session.default_fix.draw()
                
        elif self.phase == 3:
            # response
            self.session.green_fix.draw()
            if self.session.photodiode_check:
                self.black_square.draw()

        elif self.phase == 4:
            # response feedback
            # if no response, draw black fix, otherwise default
            if self.session.photodiode_check:
                self.black_square.draw()
            
            if self.parameters['response_given']:
                
                if self.session.settings['task']['advance_directly']:
                    # stop phase directly
                    self.stop_phase()
                else:
                    self.session.default_fix.draw()
                
            else:
                self.session.black_fix.draw()

        elif self.phase == 5:
            if self.session.photodiode_check:
                self.black_square.draw()

            # confidence response if a response was given before
            if self.parameters['response_given']:
                self.session.blue_fix.draw()
            else:
                self.stop_phase()

        elif self.phase == 6:
            if self.session.photodiode_check:
                self.black_square.draw()
            # confidence feedback
            # if no response, draw black fix, otherwise default
            if self.parameters['confidence_response_given']:
                self.session.default_fix.draw()
            else:
                if self.parameters['response_given']:
                    self.session.black_fix.draw()
                else:
                    self.stop_phase()

    def get_events(self):

        events = event.getKeys(timeStamped=self.session.clock)
        if events:
            if 'q' in [ev[0] for ev in events]:  # specific key in settings?
                self.session.close()
                self.session.quit()
            
            event_type = 'button' # default event type for button presses outside responses

            for key, t in events:
                

                if self.phase == 0:
                    if any(key in events[0] for key in ['1', '2', '3', '4', '5', 'space']):
                    
                        self.stop_phase()

                # print(self.phase)
                if self.phase == 3:
                    event_type = 'response'
                    # print(f"t is: {t}")
                    # some logic depending on the task and response
                    if self.session.settings['task']['type'] == 'TOJ':
                        # temporal order judgement: responses are a/v first
                        if key == self.session.response_button_mapping['audio_first']:
                            self.parameters['response_given'] = True
                            
                            # pp indicated audio first
                            if self.parameters['order'] == 'AV':
                                # correct
                                self.parameters['correct'] = 1
                                self.parameters['response'] = 'A'
                                
                                
                                # get start of response window
                                resp_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.trial_nr) & \
                                                                                (self.session.global_log['event_type']=='response_window'), 'onset'].to_numpy()

                                RT = t - resp_onset
                                self.parameters['RT'] = RT
                                print(f"trial {self.trial_nr} rt is: {RT}, resp_onset for conf is {resp_onset}")
                                self.stop_phase()

                            else:

                                # incorrect
                                self.parameters['correct'] = 0
                                self.parameters['response'] = 'A'
                                
                                # get start of response window
                                resp_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.trial_nr) & \
                                                                                (self.session.global_log['event_type']=='response_window'), 'onset'].to_numpy()

                                RT = t - resp_onset
                                self.parameters['RT'] = RT
                                print(f"trial {self.trial_nr} rt is: {RT}, resp_onset for conf is {resp_onset}")
                                self.stop_phase()

                        elif key == self.session.response_button_mapping['visual_first']:
                            self.parameters['response_given'] = True
                            
                            # pp indicated visual first
                            if self.parameters['order'] == 'VA':
                                # correct
                                self.parameters['correct'] = 1
                                self.parameters['response'] = 'V'

                                # get start of response window
                                resp_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.trial_nr) & \
                                                                                (self.session.global_log['event_type']=='response_window'), 'onset'].to_numpy()

                                RT = t - resp_onset
                                self.parameters['RT'] = RT
                                print(f"trial {self.trial_nr} rt is: {RT}, resp_onset for conf is {resp_onset}")
                                self.stop_phase()

                            else:
                                # incorrect
                                self.parameters['correct'] = 0
                                self.parameters['response'] = 'V'

                                # get start of response window
                                resp_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.trial_nr) & \
                                                                                (self.session.global_log['event_type']=='response_window'), 'onset'].to_numpy()

                                RT = t - resp_onset
                                self.parameters['RT'] = RT
                                print(f"trial {self.trial_nr} rt is: {RT}, resp_onset for conf is {resp_onset}")
                                self.stop_phase()

                    if self.session.settings['task']['type'] == 'SJ':
                        # temporal order judgement: responses are a/v first
                        if key == self.session.response_button_mapping['synchronous']:
                            self.parameters['response_given'] = True
                            
                            # pp indicated synchonous
                            self.parameters['response'] = 'synchronous'

                            # get start of response window
                            resp_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.trial_nr) & \
                                                                            (self.session.global_log['event_type']=='response_window'), 'onset'].to_numpy()

                            RT = t - resp_onset
                            self.parameters['RT'] = RT
                            print(f"trial {self.trial_nr} rt is: {RT}, resp_onset for conf is {resp_onset}")
                            self.stop_phase()

                        elif key == self.session.response_button_mapping['asynchronous']:
                            self.parameters['response_given'] = True
                            
                            # pp indicated asynchronous
                            self.parameters['response'] = 'asynchronous'

                    
                            # get start of response window
                            resp_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.trial_nr) & \
                                                                            (self.session.global_log['event_type']=='response_window'), 'onset'].to_numpy()

                            RT = t - resp_onset
                            self.parameters['RT'] = RT
                            print(f"trial {self.trial_nr} rt is: {RT}, resp_onset for conf is {resp_onset}")
                            self.stop_phase()
                
                elif self.phase == 5:
                    event_type = 'confidence_response'
                    if key == self.session.response_button_mapping['low_confidence']:
                        self.parameters['confidence_response_given'] = True
                        self.parameters['response'] = 'L'


                    elif key == self.session.response_button_mapping['high_confidence']:
                        self.parameters['confidence_response_given'] = True
                        self.parameters['response'] = 'H'
                                
                    # get start of response window
                    resp_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.trial_nr) & \
                                                                    (self.session.global_log['event_type']=='confidence'), 'onset'].to_numpy()

                    RT = t - resp_onset
                    self.parameters['RT'] = RT
                    print(f"trial {self.trial_nr} rt is: {RT}, resp_onset for conf is {resp_onset}")
                    


                    self.stop_phase()

                # log:
                idx = self.session.global_log.shape[0]
                self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                self.session.global_log.loc[idx, 'onset'] = t
                self.session.global_log.loc[idx, 'event_type'] = event_type
                self.session.global_log.loc[idx, 'phase'] = self.phase
                self.session.global_log.loc[idx, 'key'] = key

                # for param, val in self.parameters.items():
                    # self.session.global_log.loc[idx, param] = val
                for param, val in self.parameters.items():  # add parameters to log
                    if type(val) == np.ndarray or type(val) == list:
                        for i, x in enumerate(val):
                            self.session.global_log.loc[idx, param+'_%4i'%i] = x 
                    else:       
                        self.session.global_log.loc[idx, param] = val

                if self.eyetracker_on:  # send msg to eyetracker
                    msg = f'start_type-{event_type}_trial-{self.trial_nr}_phase-{self.phase}_key-{key}_time-{t}'
                    self.session.tracker.sendMessage(msg)

                if key != self.session.mri_trigger:
                    self.last_resp = key
                    self.last_resp_onset = t
        if self.session.photodiode_check and (not self.recorded) and (self.phase == 5):

            front_rec = self.session.mic_front.getRecording()
            back_rec = self.session.mic_back.getRecording()
            print(f'front shape: {front_rec.samples.shape}\n\
                    back shape {back_rec.samples.shape}')

            self.session.recordings.append([front_rec.samples, back_rec.samples])
            # self.session.conditions.append() 
            self.session.trial_type.append(self.parameters["order"])
            self.session.soa_durations.append(self.parameters["soa"])
            self.session.recording_durations_front.append(front_rec.duration) 
            self.session.recording_durations_back.append(back_rec.duration) 

            # self.session.delta_peaks.append()
            # self.session.n_peaks_found.append()

            self.recorded = True

        return events

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
            self.session.on_phase_frame = 0

            # Start loading in next trial during this phase (if not None)
            if self.load_next_during_phase == self.phase:
                self.load_next_trial(phase_dur)

            if self.timing == 'seconds':
                # Loop until timer is at 0!
                self.session.timer.add(phase_dur)
                while self.session.timer.getTime() < 0 and not self.exit_phase and not self.exit_trial:
                    self.draw()
                    if self.draw_each_frame:
                        self.session.win.flip()
                        self.session.nr_frames += 1
                    self.get_events()
            else:
                # Loop for a predetermined number of frames
                # Note: only works when you're sure you're not 
                # dropping frames
                for _ in range(phase_dur):

                    if self.exit_phase or self.exit_trial:
                        break

                    if self.session.photodiode_check:
                        if (self.phase == 2) and (_ == 0):
                            # starting mics at start of phase2 
                            self.session.mic_front.start()
                            self.session.mic_back.start()
                        if (self.phase == 3) and (_ == 0):
                            # closing mics at start of phase3 
                            self.session.mic_front.stop()
                            self.session.mic_back.stop()

                            # print(f'time elapsed between start of mics:\
                            #       {self.session.mic_front.stop()[0] - self.session.mic_back.stop()[0]}')

                    self.draw()
                    t = self.session.win.flip()
                    if (self.phase == 2) and (_ == 1):
                        print(t)
                    self.get_events()
                    self.session.nr_frames += 1
                    self.session.on_phase_frame += 1


            if self.exit_phase:  # broke out of phase loop
                self.session.timer.reset()  # reset timer!
                self.exit_phase = False  # reset exit_phase
            if self.exit_trial:
                self.session.timer.reset()
                break

            self.phase += 1  # advance phase

class TempIntSession(PylinkEyetrackerSession):
    """ Simple session with x trials. """

    def __init__(self, output_str, output_dir=None, settings_file=None, n_trials=10,
                 photodiode_check = False, eyetracker_on=True, debug = False):
        """ Initializes TestSession object. """
        self.n_trials = n_trials
        super().__init__(output_str, output_dir=output_dir,
                         settings_file=settings_file, eyetracker_on=eyetracker_on)
        
        self.green_fix = Circle(self.win, radius=.075, edges = 100, lineWidth=0)
        self.green_fix.setColor((0, 128, 0), 'rgb255')
        self.black_fix = Circle(self.win, radius=.075, edges = 100, lineWidth=0)
        self.black_fix.setColor((0, 0, 0), 'rgb255')
        self.blue_fix = Circle(self.win, radius=.075, edges = 100, lineWidth=0)
        self.blue_fix.setColor((0, 171, 240), 'rgb255')

        # get stimulus params
        self.conds = self.settings['stimuli']['stim_conds']
        self.n_conds = len(self.conds)
        self.n_repeats_per_block = self.settings['stimuli']['n_repeats_per_block']
        self.n_blocks = self.settings['stimuli']['n_blocks']

        self.stim_dur_vis = self.settings['stimuli']['stim_dur_vis']
        self.stim_dur_aud = self.settings['stimuli']['stim_dur_aud']
        self.stim_onset_asynch = self.settings['stimuli']['stim_onset_asynch']
        self.conds_tuple = list(itertools.product(self.conds, self.stim_onset_asynch))
        
        # make fix big for movie
        if self.settings['stimuli']['screenshot']:
            self.green_fix.setSize(5)
            self.black_fix.setSize(5)
            self.blue_fix.setSize(5)
            self.default_fix.setSize(5)
        
        try:
            self.conds_tuple.remove(('AV',0)) # removing one of the simultaneous conds, as we only need one
        except ValueError:
            print("No (AV, 0) condition specified. Please check if this is intended.")

        # make sound
        self.sound = sound.Sound(secs = self.stim_dur_aud/120)

        # handle photodiode
        self.photodiode_check = True if photodiode_check else False 
        if self.photodiode_check == True:

            # frontside HDA Intel PCA
            self.mic_front=Microphone(device=0, streamBufferSecs=1.0, channels=2)
            # backside HDA Creative, rightmost jack
            self.mic_back=Microphone(device=6, streamBufferSecs=1.0, channels=2)

            self.recordings = []
            self.trial_type = [] 
            self.recording_durations_front = [] 
            self.recording_durations_back = [] 
            self.soa_durations = []


        # set phase frame counter
        self.on_phase_frame = 0
        # set phase names for logs 
        if self.settings['task']['confidence']:
            self.phase_names = ['start_screen','ITI','stim','response_window', 'response_feedback', 'confidence', 'confidence_feedback']
        else:
            self.phase_names = ['start_screen','ITI','stim','response_window', 'response_feedback']

        if n_trials is None:
            self.n_trials_block = self.n_repeats_per_block * len(self.conds_tuple)
            self.n_trials = self.n_trials_block * self.n_blocks

        self.debug = debug
        # setting a debug message
        if self.debug:
            self.debug_message = TextStim(self.win, text = "debug text", pos = (6.0,5.0), height = .3,
                                       opacity = .5) 

        # make response button mapping
        if self.settings['task']['type'] == 'SJ':
            # synchrony judgement mapping
            self.response_button_mapping = {'synchronous' : self.settings['task']['response_keys'][0],
                                        'asynchronous' : self.settings['task']['response_keys'][1],
                                        'low_confidence': self.settings['task']['confidence_response_keys'][0],
                                        'high_confidence': self.settings['task']['confidence_response_keys'][1],
                                        }
        elif self.settings['task']['type'] == 'TOJ':
            self.response_button_mapping = {'audio_first' : self.settings['task']['response_keys'][0],
                                        'visual_first' : self.settings['task']['response_keys'][1],
                                        'low_confidence': self.settings['task']['confidence_response_keys'][0],
                                        'high_confidence': self.settings['task']['confidence_response_keys'][1],
                                        }
        else:
            raise ValueError(f"{self.settings['task']['type']} is not supported as a task")
            
        # paths to textures:
        if self.settings['stimuli']['tex_type'] == 'snakes-new':
            # assuming snake density of 4
            self.texture_paths = list(glob.glob(f"../textures/{self.settings['stimuli']['tex_type']}/density-4/*"))
        else:
            self.texture_paths = list(glob.glob(f"../textures/{self.settings['stimuli']['tex_type']}/*"))
        # randomize
        random.shuffle(self.texture_paths)

        # init result lists
        self.response_times = []
        self.target_times = []
        self.target_times_f = []

    def create_trials(self, durations=None, timing='frames'):
        # make trial parameters (AV/VA and SOA)
        # all_trial_parameters = self.conds_tuple * self.n_repeats
        # init trial params list
        all_trial_parameters = []

        # fill it up, making sure blocks are balanced TODO balance in 'miniblocks' a n_conditons
        # for block in range(self.n_blocks):
        #     # shuffle inplace
        #     random.shuffle(all_trial_parameters_block)
        #     # add a copy 
        #     all_trial_parameters += all_trial_parameters_block

        if self.settings['stimuli']['randomization'] == 'block':
            all_trial_parameters_block = self.conds_tuple * self.n_repeats_per_block
            
            for block in range(self.n_blocks):
                # shuffle inplace
                random.shuffle(all_trial_parameters_block)
                # add a copy 
                all_trial_parameters += all_trial_parameters_block
                
        elif self.settings['stimuli']['randomization'] == 'cond':
            for i in range(self.n_blocks * self.n_repeats_per_block):
                params_miniblock = self.conds_tuple.copy()
                random.shuffle(params_miniblock) # randomize
                all_trial_parameters += params_miniblock
        
        print(f"made durs {all_trial_parameters}")

        p0_durs = [0] * self.n_trials # instructions/break screen, only to be triggered at start and end of block

        # jits = np.arange(54, 79) # jittered times between 54 and 79 frames, corr. to 425 - 600 ms, as in Yankieva
        jits = np.arange(130, 170) #

        p1_durs = [int(jit) for jit in np.random.choice(jits, self.n_trials)]
        p2_durs = [self.stim_dur_aud * 2 + params[-1] for params in all_trial_parameters]
        p3_durs = [3 * 120] * self.n_trials # fixed duration for responses, 144 frames are 1.2 s
        p4_durs = [30] * self.n_trials # fixed blank for response feedback, 30 frames are 250 ms
        p5_durs = [3 * 120] * self.n_trials # fixed duration for confidence, 144 frames are 1.2 s
        p6_durs = [30] * self.n_trials # fixed blank for confidence feedback, 30 frames are 250 ms

        if self.photodiode_check:
            p1_durs = [int(jit) for jit in np.random.choice(jits, self.n_trials)]
            p2_durs = [self.stim_dur_aud * 2 + params[-1] for params in all_trial_parameters]
            p3_durs = [10] * self.n_trials # fixed duration for responses, 144 frames are 1.2 s
            p4_durs = [10] * self.n_trials # fixed blank for response feedback, 30 frames are 250 ms
            p5_durs = [10] * self.n_trials # fixed duration for confidence, 144 frames are 1.2 s
            p6_durs = [10] * self.n_trials # fixed blank for confidence feedback, 30 frames are 250 ms
 

        self.trials = []
        if self.settings['task']['confidence']:
        
            durations = list(zip(p0_durs, p1_durs, p2_durs, p3_durs, p4_durs, p5_durs, p6_durs))
        else:
            durations = list(zip(p0_durs, p1_durs, p2_durs, p3_durs, p4_durs))

        for trial_nr in range(self.n_trials):

            # set time for p0 of first trial of each block to large to trigger tutorial screen/block pause
            if trial_nr % self.n_trials_block == 0:
                # turn to list to change duration of phase 0
                durations[trial_nr] = list(durations[trial_nr])
                durations[trial_nr][0] = 432000 # 432000 frames correspond to an hour
                # changing also duration of phase 1 to be a bit longer to accomodate participant
                durations[trial_nr][1] += 2*120  


            self.trials.append(
                TempIntTrial(session=self,
                          trial_nr=trial_nr,
                          phase_durations=durations[trial_nr],
                          phase_names = self.phase_names,
                          txt='Trial %i' % trial_nr,
                          verbose=False,
                          parameters=dict(order=all_trial_parameters[trial_nr][0],
                                          soa=all_trial_parameters[trial_nr][1]),
                          timing=timing)
            )

    def close(self):
        """'Closes' experiment. Should always be called, even when10
        experiment is quit manually (saves onsets to file)."""

        if self.closed:  # already closed!
            return None

        self.win.callOnFlip(self._set_exp_stop)
        self.win.flip()
        self.win.recordFrameIntervals = False

        print(f"\nDuration experiment: {self.exp_stop:.3f}\n")

        if not op.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        self.global_log = pd.DataFrame(self.global_log).set_index("trial_nr")
        self.global_log["onset_abs"] = self.global_log["onset"] + self.exp_start

        # Only non-responses have a duration
        nonresp_idx = ~self.global_log.event_type.isin(["response", "trigger", "pulse", "confidence_response", "button"])
        last_phase_onset = self.global_log.loc[nonresp_idx, "onset"].iloc[-1]
        dur_last_phase = self.exp_stop - last_phase_onset
        durations = np.append(
            self.global_log.loc[nonresp_idx, "onset"].diff().values[1:], dur_last_phase
        )
        self.global_log.loc[nonresp_idx, "duration"] = durations

        # Same for nr frames
        nr_frames = np.append(
            self.global_log.loc[nonresp_idx, "nr_frames"].values[1:], self.nr_frames
        )
        self.global_log.loc[nonresp_idx, "nr_frames"] = nr_frames.astype(int)

        # Round for readability and save to disk
        self.global_log = self.global_log.round(
            {"onset": 5, "onset_abs": 5, "duration": 5}
        )
        f_out = op.join(self.output_dir, self.output_str + "_events.tsv")
        self.global_log.to_csv(f_out, sep="\t", index=True)

        if self.settings['stimuli']['screenshot']:
            print('saving movie')
            self.win.saveMovieFrames('movie.tif')

        # Create figure with frametimes (to check for dropped frames)
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(self.win.frameIntervals)
        ax.axhline(1.0 / self.actual_framerate, c="r")
        ax.axhline(
            1.0 / self.actual_framerate + 1.0 / self.actual_framerate, c="r", ls="--"
        )
        ax.set(
            xlim=(0, len(self.win.frameIntervals) + 1),
            xlabel="Frame nr",
            ylabel="Interval (sec.)",
            ylim=(-0.01, 0.125),
        )
        fig.savefig(op.join(self.output_dir, self.output_str + "_frames.pdf"))

        if self.mri_simulator is not None:
            self.mri_simulator.stop()

        # create results df
        results = {'target_times_s' : self.target_times,
                   'target_times_f' : self.target_times_f,
                   'response_times_s' : self.response_times,
                   'response_times_f' : [int(round(time * 120)) for time in self.response_times],
                   }
        
        results_df = pd.DataFrame(results)
        results_df['response_diff_f'] = [frames_target - frames_resp for frames_target, frames_resp in zip(results_df['target_times_f'], results_df['response_times_f'])]

        self.results = results_df

        # create photodiode results
        if self.photodiode_check:
            results_pd = {'recordings' : self.recordings,
                    'trial_type' : self.trial_type,
                    'recording_durations_front' : self.recording_durations_front,
                    'recording_durations_back' : self.recording_durations_back,
                    'soa_durations': self.soa_durations
                    }
            
            results_pd_df = pd.DataFrame(results_pd)
            results_pd_df.to_pickle(op.join(self.output_dir, self.output_str + 'photodiode_test_results.pkl'))

        self.win.close()
        
        if self.eyetracker_on:
            self.stop_recording_eyetracker()
            self.tracker.setOfflineMode()
            core.wait(.5)
            f_out = op.join(self.output_dir, self.output_str + '.edf')
            self.tracker.receiveDataFile(self.edf_name, f_out)
            self.tracker.close()
        
        self.closed = True

    def run(self):
        """ Runs experiment. """
        if self.eyetracker_on:
            self.calibrate_eyetracker()
            self.start_experiment()
            self.start_recording_eyetracker()
        else:
            self.start_experiment()

        for trial in self.trials:
            trial.run()

        self.close()  # contains tracker.stopRecording()

    def post_process(self, framerate = 1/120):
        ## post-processing
        # load data
        # data_path = op.join(results_folder, output_str + "_events.tsv")
        data_path = op.join(self.output_dir, self.output_str + "_events.tsv")
        df = pd.read_csv(data_path, sep='\t')

        # process
        # init result df lists, to become columns
        soas = [] # stimulus onset asynchrony
        resps = [] # responses
        which_first = [] # A or V first?

        for trial in df.trial_nr.unique():    
                # index trial
                trial_df = df[df.trial_nr == trial]
                soa = trial_df.soa.iloc[0]

                if self.settings['task']['type'] == 'SJ':
                    # process SJ
                    try:
                        # getting response in response period phase 3
                        response = trial_df[(trial_df.event_type == 'response') & (trial_df.phase == 3)].response.values[0]
                    except IndexError:
                        response = None

                    # synchonous is coded as 1
                    response = 1 if response == 'synchronous' else 0
                    resps.append(response)

                else:
                    # process TOJ
                    try:
                        # getting response in response period phase 3
                        response_correct = trial_df[(trial_df.event_type == 'response') & (trial_df.phase == 3)].correct.values[0]
                    except IndexError:
                        response_correct = None

                    # correct is coded as 1
                    resps.append(response_correct)

                first = trial_df.order.iloc[0][0] # which stimulus type was first, derived from tial type 'AV'/'VA'
                soa = soa if first == 'A' else -soa # making soas for visual first negative

                soas.append(soa)
                which_first.append(first)
        
        # make df
        resps_df = pd.DataFrame({'soa_f' : soas, 'neg_soa_f': [-soa for soa in soas], 'soa_ms': [soa*framerate*1000 for soa in soas], 
                                'neg_soa_ms': [-soa*framerate*1000 for soa in soas],
                                'response' : resps, 'first':which_first})

        # flipped response coding for V-first trials
        comb_soas = []
        responses_flipped = []

        for i, row in resps_df.iterrows():
            
            if row['first'] == 'V': 
                comb_soas.append(row['neg_soa_f'])
                # flip response
                if row['response'] == 0.0:
                    responses_flipped.append(1.0)
                else:
                    responses_flipped.append(0.0)

            else:
                comb_soas.append(row['soa_f'])
                responses_flipped.append(row['response'])
                
        resps_df['comb_soa_f'] = comb_soas
        resps_df['comb_soa_ms'] = [comb_soa *framerate*1000 for comb_soa in comb_soas]
        resps_df['responses_flipped'] = responses_flipped
        
        if self.settings['task']['type'] == 'SJ':
            mean_response = resps_df.groupby(['subject', 'soa_ms', 'session'])['response'].mean().reset_index()
            model = gaussian
            init = {'A': 1,'$\mu$':0, '$\sigma$':200}

        else:
            mean_response = resps_df.groupby(['subject', 'soa_ms', 'session'])['responses_flipped'].mean().reset_index()
            model = cumulative_normal_lapse
            init = {'$\mu$':0.02,'$\sigma$':.005, '$\lambda$': .05}
        
        # Rename columns for clarity
        mean_response.columns = ['subject', 'soa_ms', 'session', 'mean_response']

        # fit and plot
        _, fig, ax = quick_fit(mean_response.soa_ms, mean_response.mean_response,
                                model, init = init, plot = True)

        # title
        fig.suptitle(self.output_str)
        # show
        plt.show()
        # save
        plt.savefig(op.join(self.output_dir, self.output_str + "_quickfit.png"), fig)


if __name__ == '__main__':

    subject = sys.argv[1]
    sess =  sys.argv[2]
    task = 'TempInt' # different settings -> now implemented as saving the actual settings
    run = sys.argv[3] # which run    
    output_str = f'sub-{subject}_sess-{sess}_task-{task}_run-{run}'
    # print(output_str)

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

    session = TempIntSession(output_str, output_dir = results_folder,  eyetracker_on=False,
                              n_trials=None, photodiode_check = False, settings_file='settings_TempInt.yml', debug=False)

    # diagnostic printouts
    # print(session.n_conds)
    # print(session.n_repeats)
    # print(session.stim_onset_asynch)
    # print(session.n_trials)
    # print(session.conds_tuple)
    # print(sound.Sound)

    session.create_trials()
    session.run()
    # print(session.results)
    
    # results_out = f'{results_folder}/{output_str}_results.csv'
    # session.results.to_csv(results_out, index = False)
    # print(results_out)




