from exptools2.core import PylinkEyetrackerSession
from exptools2.core import Trial
from exptools2.stimuli import create_circle_fixation

from psychopy.visual import TextStim, ImageStim, Circle
from psychopy import event
from psychopy import visual, core

import random
import pyglet
import pandas as pd
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob

class TemRepTrial(Trial):
    """ Simple trial with text (trial x) and fixation. """
    def __init__(self, session, trial_nr, phase_durations, txt=None, **kwargs):
        super().__init__(session, trial_nr, phase_durations, **kwargs)
        self.txt = TextStim(self.session.win, txt) 
        # get texture        
        self.img = ImageStim(self.session.win, self.session.texture_paths[trial_nr%len(self.session.texture_paths)], #size = 10,
                             mask = 'raisedCos', maskParams = {'fringeWidth':0.2}) # proportion that will be blurred
        self.logged_response = False # flag for checking if response got logged
        self.missed_response = False # flag for checking if pp pressed response key

        if trial_nr == 0:
            # set intro message
            # intro_text = 'TEST'
            self.intro = ImageStim(self.session.win, 'assets/intructions_screen_TempRep.png', units='pix', size = [int(1920 * .95), int(1080*.95)])
        elif self.trial_nr % self.session.n_trials_block==0:
            pause_text = f"Great, you did {self.trial_nr} of {self.session.n_trials} trials.\nYou can rest a little now, but try to keep your head stable.\nContinue with any button if you're ready."
            self.pause_message = visual.TextStim(self.session.win, pos=[0,0], text= pause_text, color = (1.0, 1.0, 1.0), height=0.5, font='Arial', wrapWidth=850)


    def draw_flip(self):
        """ Draws stimuli """

        # make movie of a trial or more
        # if (self.trial_nr == 1) and self.session.settings['stimuli']['screenshot']:
        if self.session.settings['stimuli']['screenshot']:
        
            # print('getting Movie frame')
            self.session.win.getMovieFrame()

        if self.phase == 0:

            self.session.white_fix.draw()
            if self.trial_nr == 0:
                # first trial, intro screen
                self.intro.draw()

                self.session.win.flip()
                events = self.get_events()
                if events:
                    if any(key in events[0] for key in ['space', 'return']):
                        
                        self.stop_phase()

            elif self.trial_nr % self.session.n_trials_block==0:
                self.pause_message.draw()
                self.session.win.flip()
                events = self.get_events()
                if events:
                    if any(key in events[0] for key in ['1', '2', '3', '4', '5', 'space', 'return']):
                        # break out of phase upon key press to continue
                        self.stop_phase()
        
        elif self.phase == 1:
            # jittered blank
            self.session.white_fix.draw()
            self.session.win.flip()
            self.get_events()

        elif self.phase == 2:
            # show stimulus for some time
            self.img.draw()
            self.session.white_fix.draw()
            self.session.win.flip()
            self.get_events()

        elif self.phase == 3:
            # fixed blank
            # self.session.white_fix.draw()
            self.session.white_fix.draw()
            self.session.win.flip()
            self.get_events()
            event.clearEvents()
          
        elif self.phase == 4:
            # collect answer
            # on keypress show the stimulus
            # if space gets pressed, we count the time

            frame_count = 0

            events = self.get_events()

            if events:
                if ('space' in events[0]) or ('2' in events[0]):

                    # start timer
                    self.session.response_timer.reset()
                    # print(f"key pressed, started timer, {self.session.response_timer.getTime()}")

                    # start while loop that will run as long as the key stays pressed
                    key_pressed = True

                    while key_pressed and (self.session.response_timer.getTime() < 15.0):
                        # if key is pressed, we draw the stimulus
                        if (self.session.keyboard[self.session.key.SPACE]) or (self.session.keyboard[self.session.key._2]): # numbers on button bux are accessible with ._2
                            
                            if self.session.settings['stimuli']['show_stim_during_response']:
                                
                                # if (self.trial_nr == 1) and self.session.settings['stimuli']['screenshot']:
                                if self.session.settings['stimuli']['screenshot']:

                                    self.session.win.getMovieFrame()
                                # draw stimulus
                                self.img.draw()
                                # green fix on top
                                self.session.green_fix.draw()

                            else:
                                self.session.white_fix.draw()

                            # counting frames, helpful for debugging
                            frame_count += 1 
                            # flip
                            self.session.win.flip()

                        else:
                            # if key is released, key_pressed will be set to false, exiting the while loop
                            key_pressed = False

                    # getting the passed time
                    response_time = self.session.response_timer.getTime()

                    # saving passed time, target time
                    self.session.response_times.append(response_time)
                    self.session.target_times_f.append(self.phase_durations[2])
                    self.session.target_times.append(self.phase_durations[2]/120)

                    # into global_log
                    idx = self.session.global_log.shape[0] - 1 # -1 since the row we need already exists, just needs duration and nr frames

                    self.session.global_log.loc[idx, 'duration'] = response_time
                    self.session.global_log.loc[idx, 'nr_frames'] = int(np.round(response_time*120))
                    self.logged_response = True

                    # printout comparing passed time with counted frames
                    print(f"trial nr {self.trial_nr}; target: {self.phase_durations[2] / 120:.2f}s; response: {response_time:.2f}s")
                    
                    # exit phase
                    event.clearEvents() # event logging was done manually above, just clearing the event buffer before next phase 
                    self.exit_phase = True

            # as long as nothing has been pressed, a green fix ecourages pressing the button
            else:
                # green fix
                self.session.green_fix.draw()

            # print(f'flipped {i}') # debug message    
            self.session.win.flip()
            # pass

        else: # p5
            if self.logged_response == False:
                # time ran out
                # give feedback that pp was too slow
                # log that there is no answer as np.NaN in results lists
                self.session.response_times.append(np.NaN)
                self.session.target_times_f.append(self.phase_durations[2])
                self.session.target_times.append(self.phase_durations[2]/120)
                # global log
                self.logged_response = True
                self.missed_response = True
            
            if self.missed_response == True:
                self.session.black_fix.draw()
            else:
                self.session.white_fix.draw()
            
            # if (self.trial_nr == 1) and self.session.settings['stimuli']['screenshot']:
            if self.session.settings['stimuli']['screenshot']:
                self.session.win.getMovieFrame()

            # optional: black fix to indicate ITI
            # self.session.black_fix.draw()
            self.session.win.flip()
            self.get_events()



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
                    
                    self.draw_flip()
                    # self.draw()
                    # self.session.win.flip()
                    # self.get_events()
                    self.session.nr_frames += 1

            if self.exit_phase:  # broke out of phase loop
                self.session.timer.reset()  # reset timer!
                self.exit_phase = False  # reset exit_phase
            if self.exit_trial:
                self.session.timer.reset()
                break

            self.phase += 1  # advance phase

class TemRepSession(PylinkEyetrackerSession):
    """ Simple session with x trials. """

    def __init__(self, output_str, output_dir=None, settings_file=None, n_trials=10, eyetracker_on=True):
        """ Initializes TestSession object. """
        self.n_trials = n_trials
        super().__init__(output_str, output_dir=output_dir,
                         settings_file=settings_file, eyetracker_on=eyetracker_on)
        dot_size = self.settings['stimuli']['dot_size']
        self.green_fix = Circle(self.win, radius=dot_size, edges = 100, lineWidth=0)
        self.green_fix.setColor((0, 128, 0), 'rgb255')
        self.black_fix = Circle(self.win, radius=dot_size, edges = 100, lineWidth=0)
        self.black_fix.setColor((0, 0, 0), 'rgb255')
        self.white_fix = Circle(self.win, radius=dot_size, edges = 100, lineWidth=0)
        self.white_fix.setColor((255, 255, 255), 'rgb255')

        ## for screenshots, make fix bigger
        # if self.settings['stimuli']['screenshot']:
        #     self.green_fix.setSize(5)
        #     self.black_fix.setSize(5)
        #     self.white_fix.setSize(5)

        # keyboard workaround
        self.key = pyglet.window.key
        self.keyboard = self.key.KeyStateHandler()
        self.win.winHandle.push_handlers(self.keyboard)
        
        # response timer
        self.response_timer = core.Clock()
        
        # get stimulus params
        self.conds = self.settings['stimuli']['stim_conds']
        self.n_conds = len(self.conds)
        self.n_repeats = self.settings['stimuli']['n_repeats']
        self.n_blocks = self.settings['stimuli']['n_blocks']

        # paths to textures:
        if self.settings['stimuli']['tex_type'] == 'snakes-new':
            # assuming snake density of 4
            self.texture_paths = list(glob.glob(f"textures/{self.settings['stimuli']['tex_type']}/density-4/*"))
        else:
            self.texture_paths = list(glob.glob(f"textures/{self.settings['stimuli']['tex_type']}/*"))
        # randomize
        random.shuffle(self.texture_paths)
        if n_trials is None:
            self.n_trials_block = self.n_conds * self.n_repeats
            self.n_trials = self.n_trials_block * self.n_blocks

        # init result lists
        self.response_times = []
        self.target_times = []
        self.target_times_f = []

    def create_trials(self, durations=None, timing='frames'):

        # for block in range(self.n_blocks):
        # set up trial times blockwise to ensure balanced blocks
        p0_durs = [0] * self.n_trials # instructions/break screen, only to be triggered at start and end of block
        jits = np.arange(72, 120) 
        p1_durs = [int(jit) for jit in np.random.choice(jits, self.n_trials)] # jittered blank part of iti
        
        p2_durs = []

        if self.settings['stimuli']['randomization'] == 'block':
            for block in range(self.n_blocks):
                p2_durs_block = self.conds * self.n_repeats # random durations showing stim
                random.shuffle(p2_durs_block) # randomize
                p2_durs += p2_durs_block
        
        elif self.settings['stimuli']['randomization'] == 'cond':
            for i in range(self.n_blocks * self.n_repeats):
                p2_durs_miniblock = self.conds.copy()
                random.shuffle(p2_durs_miniblock) # randomize
                p2_durs += p2_durs_miniblock
        print(f"made durs {p2_durs}")

        p3_durs = [60] * self.n_trials # 500 ms/60 frames blank
        p4_durs = [int(1.5 * 120)] * self.n_trials #  1.5 seconds to start answer
        p5_durs = [36] * self.n_trials # response feedback 30 frames = 250 ms
        
        # end with a bit of a buffer, 2 seconds
        p5_durs[-1] = 240

        self.trials = []
        durations = list(zip(p0_durs, p1_durs, p2_durs, p3_durs, p4_durs, p5_durs))
        # print(len(durations))
        for trial_nr in range(self.n_trials):

            # set time for p0 of first trial of each block to large to trigger tutorial screen/block pause
            if trial_nr % self.n_trials_block == 0:
                # turn to list to change duration of phase 0
                # print(trial_nr)
                durations[trial_nr] = list(durations[trial_nr])
                durations[trial_nr][0] = 432000 # 432000 frames correspond to an hour
                # phase 1 slightly longer 
                durations[trial_nr][1] += 2 * 120 


            self.trials.append(
                TemRepTrial(session=self,
                          trial_nr=trial_nr,
                          phase_durations=durations[trial_nr],
                          txt='Trial %i' % trial_nr,
                          verbose=False,
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
        nonresp_idx = ~self.global_log.event_type.isin(["response", "trigger", "pulse"])
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

        # create results df
        results = {'target_times_s' : self.target_times,
                   'target_times_f' : self.target_times_f,
                   'response_times_s' : [time if time is not np.NaN else -1 for time in self.response_times], # -1 to indicate trials without response
                   'response_times_f' : [int(round(time * 120)) if time is not np.NaN else -1 for time in self.response_times], # -1 to indicate trials without response
                   }

        results_df = pd.DataFrame(results)
        # np.NaNs for trials without response
        results_df['response_diff_f'] = [frames_target - frames_resp if frames_resp != -1 else np.NaN for frames_target, frames_resp in zip(results_df['target_times_f'], results_df['response_times_f'])]

        self.results = results_df
        results_out = f'{self.output_dir}/{self.output_str}_results.csv'

        # print(self.target_times_f)
        # print(self.response_times)
        
        print(f"saving results at {results_out}")

        self.results.to_csv(results_out, index = False)

        if self.settings['stimuli']['screenshot']:
            print('saving movie')
            # self.win.saveMovieFrames('movie.tif')
            self.win.saveMovieFrames('movie.mp4')


        self.win.close()


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

    def post_process(self):
        # perform cleaning
        # check for -1's in response_times_s
        drop_idx = np.where(self.results.response_times_f == -1)[0]
        self.results.drop(drop_idx)
        print(f"Post-processing: removed {len(drop_idx)} trials without answer")

        times = []
        unique_response_times = np.sort(self.results.target_times_f.unique())

        for target_time in unique_response_times:
        #     print(target_time)
            times.append(self.results[self.results.target_times_f == target_time].response_times_f)
            
        # box_data = np.vstack(times).T
        plt.style.use('seaborn-talk')
        fig, axs = plt.subplots(1, 2, figsize = (12,6))

        x_positions = [16, 32, 64] 
        axs[0].plot([16, 32, 64], [16, 32, 64], marker = 'o', color = 'black', linestyle = '--')

        ax_range = axs[0].get_xlim()[1] - axs[0].get_xlim()[0]
        widths = ax_range/5

        # axs[0].boxplot(box_data[:,:3], positions = unique_response_times[:3], widths = widths,)
        axs[0].boxplot(times[:3], positions = unique_response_times[:3], widths = widths,)

        axs[0].set_xticks(x_positions)
        axs[0].set_yticks(x_positions)
        axs[0].set_ylim(0, 200)

        ticks_in_s = [np.round(x/120, 3) for x in axs[0].get_xticks()]
        axs[0].set_xticklabels(ticks_in_s)
        axs[0].set_yticklabels(ticks_in_s)

        x_positions = [128, 256, 512] 
        axs[1].plot([128, 256, 512], [128, 256, 512], marker = 'o', color = 'black', linestyle = '--')

        ax_range = axs[1].get_xlim()[1] - axs[1].get_xlim()[0]
        widths = ax_range/5

        axs[1].boxplot(times[3:], positions = unique_response_times[3:], widths = widths)

        axs[1].set_xticks(x_positions)
        axs[1].set_yticks(x_positions)
        axs[1].set_ylim(0, 700)

        ticks_in_s = [np.round(x/120) for x in axs[1].get_xticks()]
        axs[1].set_xticklabels(ticks_in_s)
        axs[1].set_yticklabels(ticks_in_s)

        for ax in axs:
            ax.set_xlabel('target time [s]')
            ax.set_ylabel('response time [s]')

        plt.tight_layout()

        plt.show()

        out_path = os.path.join(self.output_dir, f'{self.output_str}_quick_results.png')
        os.makedirs(self.output_dir, exist_ok = True)

        plt.savefig(out_path, facecolor = 'white')


if __name__ == '__main__':

    subject = sys.argv[1]
    sess =  sys.argv[2]
    task = 'TempRep' # different settings -> now implemented as saving the actual settings
    run = sys.argv[3] # which run    
    output_str = f'sub-{subject}_sess-{sess}_task-{task}_run-{run}'
    results_folder = f'{task}_pilot/sub-{subject}/ses-{sess}'
    # print(results_folder) 

    # Check if the directory already exists
    if not os.path.exists(results_folder):
        # Create the directory
        os.makedirs(results_folder)
        print("results_folder created successfully!")
    else:
        print("results_folder already exists!")

    session = TemRepSession(output_str, output_dir = results_folder,  eyetracker_on=False, n_trials=None, settings_file='settings_TemRep.yml')

    session.create_trials()
    session.run()
    # session.post_process()
    # results_out = f'{results_folder}/{output_str}_results.csv'
    # session.results.to_csv(results_out, index = False)
    # print(results_out)

