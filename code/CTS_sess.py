# imports exptools and psychopy
from exptools2.core import PylinkEyetrackerSession
from psychopy.visual import TextStim
# from psychopy.sound import Microphone
from psychopy.event import waitKeys, getKeys
from psychopy import core
from code.CTS_trial import DelayedNormTrial

# other imorts
import os.path as op
import numpy as np
import pandas as pd
from scipy.stats import norm
import glob
import os
import matplotlib.pyplot as plt
from datetime import datetime
import json

class DelayedNormSession(PylinkEyetrackerSession):
    """ CTS/DN session. """
    def __init__(self, output_str, output_dir=None, settings_file=None, n_trials=None, eyetracker_on=True, photodiode_check = False, debug = False):

        super().__init__(output_str, output_dir=output_dir, settings_file=settings_file, eyetracker_on=eyetracker_on)
        
        # load params from sequence df
        self.trial_sequence_df = pd.read_csv(self.settings['stimuli']['trial_sequence'])
        self.TR = self.settings['mri']['TR'] # TR in seconds

        self.metadata = {'settings_file' : settings_file,
                         'eyetracker': eyetracker_on}

        if n_trials is None:
            # just the length of the trial df
            self.n_trials = len(self.trial_sequence_df)
        else:
            # reduce to n_trials for debugging 
            self.n_trials = n_trials
            self.trial_sequence_df = self.trial_sequence_df[:n_trials]

        # keeping track of frame timings
        self.trialwise_frame_timings = np.zeros((96, self.n_trials))
        self.trial_frames = 0

        # fix dot color and size  
        self.fix_dot_color_idx = 0
        self.fix_dot_switch_idx = 0

        self.fix_dot_colors = ['green', 'red']
        self.default_fix.setSize(self.settings['task']['fix_dot_size'])
        self.default_fix.setColor('green') # starting color
        self.default_fix.setPos((self.settings['stimuli']['x_offset'], self.settings['stimuli']['y_offset'])) # starting color        
        # self.default_fix.setPos(self.settings['stimuli']['x_offset'],self.settings['stimuli']['y_offset']) # starting color        

        # setting up fixation task duration and timings
        self.total_TRs = self.settings['stimuli']['blank_before_trs'] + np.sum(self.trial_sequence_df.iti_TR) + self.settings['stimuli']['blank_after_trs']
        self.total_exp_duration_s = self.total_TRs * self.TR
        self.total_exp_duration_f = self.total_exp_duration_s * 120
        print(f'total TRs: {self.total_TRs}')
        # adding the triggerless should not be necessary, as the clock gets reset with start_experiment
        self.total_fix_duration = self.total_exp_duration_s
        # self.total_fix_duration = self.total_exp_duration_s + self.settings['stimuli']['triggerless_trs'] * self.TR
        self.n_hits = 0
        self.n_fas = 0
        self.effective_fix_color_switches = [] 
        self.last_fix_color_switch = None

        if self.settings['mri']['topup_scan']: 
            self.total_fix_duration += self.settings['mri']['topup_duration']
        
        print(f'total fix duration: {self.total_fix_duration}')

        if debug:
            print(f'total TRs: {self.total_TRs}')
            print(f'total Exp (s): {self.total_exp_duration_s}')
            print(f'total Exp (f): {self.total_exp_duration_f}')
            print(f'total fix (s): {self.total_fix_duration}')

        # making them in run
        # self.fix_dot_color_timings = self._make_fix_dot_color_timings()
        
        # debug flag
        self.debug = True if debug else False
        
        # setting a debug message
        if debug:
            self.debug_message = TextStim(self.win, text = "debug text", pos = (6.0,5.0), height = .3,
                                       opacity = .5) 

        # photodiode checking code
        self.photodiode_check = True if photodiode_check else False

        if photodiode_check == True:
            # only duration
            self.trial_sequence_df = self.trial_sequence_df[self.trial_sequence_df.type == 'dur']
            # quick
            self.trial_sequence_df.iti_TR = [3 for i in range(len(self.trial_sequence_df))]
            # triple
            self.trial_sequence_df = pd.concat([self.trial_sequence_df, self.trial_sequence_df, self.trial_sequence_df])

            self.mic = Microphone(streamBufferSecs=6.0)  # open the microphone
            self.recordings = {"dur" : {timing: [] for timing in [0, 2, 4, 8, 16, 32, 64]}, # hardcoded for now
                               "var" : {timing: [] for timing in [0, 2, 4, 8, 16, 32, 64]}}
            
            self.conditions = [] 
            self.trial_type = [] 
            self.recording_durations = [] 
            self.delta_peaks = [] 
            self.n_peaks_found = [] 

    def create_trials(self, timing='frames'):
        
        self.trials = []
        # TR in frames
        TR_in_frames = int(round(self.TR*120))
        # iti in TRs
        iti_TRs = self.trial_sequence_df['iti_TR']
        # iti in frames
        iti_frames = [int(iti_TR * TR_in_frames) for iti_TR in iti_TRs]

        ## phase durations of the 3 phases: prep (p0), trial (p1) and ITI (p2)
        if self.settings['stimuli']['scanner_sync']:
            # just a very large duration, as we wait for t
            prep_durations_p0 = self.n_trials * [100000] 
        else:
            # 1/2 of a TR, setting up and waiting for t # also 1.32/2 will be 79.6 frames, rounding to 80
            prep_durations_p0 = self.n_trials * [TR_in_frames//2]

        # stim_duration_p1 is the duration of a single stimulus, usually 96 frames or 800 ms for every trial
        stim_duration_p1 = self.settings['stimuli']['stim_duration']
        stim_durations_p1 = self.n_trials * [stim_duration_p1]
        # itis are the time from onset of a stimulus, while iti_durations_p2 are between the end of p1 and start of p0 so it will be ITI - 1/2 TR - 96
        iti_durations_p2 = [iti_frames_trial - stim_duration_p1 - TR_in_frames//2 for iti_frames_trial in iti_frames] 

        # make phase durations list of tuples for prep, iti, trial
        phase_durations = list(zip(prep_durations_p0, stim_durations_p1, iti_durations_p2))

        ## making stimulus arrays
        self.stim_conds = self.settings['stimuli']['stim_conds'] # frames in 120 FPS, either duration or isi times
        self.fixed_duration = self.settings['stimuli']['fixed_duration'] # fixed duration for isi trials
        self.total_duration = self.settings['stimuli']['stim_duration'] # (<800 ms in total) in exp design; 800 ms = .8*120 = 96 frames
        self._make_trial_frame_timings()

        # get paths to textures
        self.texture_paths = glob.glob(f"textures/{self.settings['stimuli']['tex_type']}/*") # get paths to textures

        # read trial_sequence_df for trial parameters
        params = [dict(trial_type = row.type,
                       stim_dur = row.cond_frames, 
                       texture_path = self.texture_paths[row.texture_id])
                  for i, row in self.trial_sequence_df.iterrows()] 

        # construct trials
        for trial_nr in range(self.n_trials):
            self.trials.append(
                DelayedNormTrial(session=self,
                          trial_nr=trial_nr,
                          phase_durations=phase_durations[trial_nr],
                          txt='Trial %i' % trial_nr,
                          parameters=params[trial_nr],
                          verbose=False,
                          timing=timing)
            )
            # debugging printout
            if self.debug:
                print(f"made trial {trial_nr} with params: {params[trial_nr]}\
                       phase duration {phase_durations[trial_nr]} and timing: {timing}")
        
        # make a dummy trial at the start
        dummy = DelayedNormTrial(session=self,
                        #   trial_nr=self.n_trials, # sets it to +1 the last trial number
                          trial_nr='dummy',
                          phase_durations=(0, 0, int(self.settings['stimuli']['blank_before_trs']* self.TR * 120) - int((self.TR* 120)//2)),
                          txt='Trial %i: Dummy' % trial_nr,
                          parameters=dict(trial_type = 'dur',
                                            stim_dur = 0, 
                                            texture_path = self.texture_paths[0]),
                          verbose=False,
                          timing=timing)
        
        # TODO remove if not needed ultimately
        # dummy_end = DelayedNormTrial(session=self,
        #             trial_nr=998,
        #             phase_durations=(0, 0, self.settings['stimuli']['dummy_trial_trs']*240),
        #             txt='Trial %i: Dummy' % trial_nr,
        #             parameters=dict(trial_type = 'dur',
        #                             stim_dur = 0, 
        #                             texture_path = self.texture_paths[0]),
        #             verbose=False,
        #             timing=timing)
        
        self.trials = [dummy] + self.trials #+ [dummy_end]
        # self.trials =  self.trials #+ [dummy_end]


    def _make_trial_frame_timings(self):
        """
        makes frame-wise sequences for stimulus presentation
        flip versions are needed for photodiode
        """
        var_duration = np.vstack([np.hstack((np.ones(stim_cond), # showing stimulus
                                             np.zeros(self.total_duration - stim_cond))) # no stimulus for the remaining frames
                                             for stim_cond in self.stim_conds])
        
        var_isi = np.vstack([np.hstack((np.ones(self.fixed_duration), # show stimulus
                                        np.zeros(stim_cond), # isi
                                        np.ones(self.fixed_duration), # show stimulus again
                                        np.zeros(self.total_duration - stim_cond - 2*self.fixed_duration))) # no stimulus for remaining frames 
                                        for stim_cond in self.stim_conds])
        
        # these dicts are integer indexable with the current number of trial frames 
        self.var_isi_dict = {dur:frames for dur, frames in zip(self.stim_conds, var_isi)}
        self.var_dur_dict = {dur:frames for dur, frames in zip(self.stim_conds, var_duration)}

        var_duration_flip = np.zeros((len(self.stim_conds), self.total_duration))
        for i in range(len(self.stim_conds)):
            #print(i)
            var_duration_flip[i, 0] = 1 # on
            var_duration_flip[i, self.stim_conds[i]] = -1 # off
            
            if self.stim_conds[i] == 0:
                var_duration_flip[i, 0] = 0
        
        var_isi_flip = np.zeros((len(self.stim_conds), self.total_duration))

        for i in range(len(self.stim_conds)):
            #print(i)
            
            if i == 0:
                var_isi_flip[i, 0] = 1 # on
                var_isi_flip[i, 2 * self.fixed_duration] = -1 # off
                
            else:
                try:
                    # fixed 16 frames
                    var_isi_flip[i, 0] = 1 # on
                    var_isi_flip[i, 0 + self.fixed_duration] = -1 # off
                    var_isi_flip[i, 0 + self.fixed_duration + self.stim_conds[i]] = 1 # on
                    var_isi_flip[i, 0 + self.fixed_duration + self.stim_conds[i] + self.fixed_duration] = -1 # off
                except IndexError:
                    continue
        
        # these dicts are integer indexable with the current number of trial frames 
        self.var_isi_dict_flip = {dur:frames for dur, frames in zip(self.stim_conds, var_isi_flip)}
        self.var_dur_dict_flip = {dur:frames for dur, frames in zip(self.stim_conds, var_duration_flip)}

        return

    def _make_fix_dot_color_timings(self, total_time=None, round_to = None):

        if total_time is None:
            total_time = self.total_fix_duration
        # Inspired by Marco's fixation task
        dot_switch_color_times = np.arange(3, total_time, self.settings['task']['color switch interval'])
        # adding randomness
        dot_switch_color_times += (2*np.random.rand(len(dot_switch_color_times))-1) # adding uniform noise [-1, 1] 
        # last one will be total time, ending it all
        dot_switch_color_times[-1] = total_time
        # round
        if round_to is not None:
            dot_switch_color_times = [np.round(time, round_to) for time in dot_switch_color_times]

        # transforming to frames 
        # dot_switch_color_times = (dot_switch_color_times*120).astype(int)

        return dot_switch_color_times


    def start_experiment(self, wait_n_triggers=None, show_fix_during_dummies=True):
        """Logs the onset of the start of the experiment.

        Parameters
        ----------
        wait_n_triggers : int (or None)
            Number of MRI-triggers ('syncs') to wait before actually
            starting the experiment. This is useful when you have
            'dummy' scans that send triggers to the stimulus-PC.
            Note: clock is still reset right after calling this
            method.
        show_fix_during_dummies : bool
            Whether to show a fixation cross during dummy scans.
        """
        self.exp_start = self.clock.getTime()
        self.clock.reset()  # resets global clock
        self.timer.reset()  # phase-timer
        self.fix_dot_switch_idx = 0 # reset switch index timer

        if self.mri_simulator is not None:
            self.mri_simulator.start()

        self.win.recordFrameIntervals = True

        if wait_n_triggers is not None:
            print(f"Waiting {wait_n_triggers} triggers before starting ...")
            n_triggers = 0
                

            while n_triggers < wait_n_triggers:
                waitKeys(keyList=[self.settings["mri"].get("sync", "t")])
                n_triggers += 1
                msg = f"\tOnset trigger {n_triggers}: {self.clock.getTime(): .5f}"
                msg = msg + "\n" if n_triggers == wait_n_triggers else msg
                print(msg)

            self.timer.reset()
    
    def switch_fix_color(self, atol = 1e-1, effective = False):
        """
        change color of default fix
        effective flag indicates whether switch happens within a trial and is logged
        """
        t = self.clock.getTime()
        # if int(t*120) in self.fix_dot_color_timings:
        if np.round(self.clock.getTime(), 2) in self.fix_dot_color_timings[self.fix_dot_switch_idx:]:
        # if np.isclose(t, self.fix_dot_color_timings[self.fix_dot_color_idx%len(self.fix_dot_color_timings)], atol = atol):

            # change color
            self.fix_dot_color_idx += 1
            self.default_fix.setColor(self.fix_dot_colors[self.fix_dot_color_idx % len(self.fix_dot_colors)])
            
            # move start index to avoid double switches 
            self.fix_dot_switch_idx += 1
            self.last_fix_color_switch = t

            if effective:
                # remove to avoid multiple occurences
                # self.fix_dot_color_timings.remove(t)
                self.effective_fix_color_switches.append(t)


    def end_experiment(self):
        """
        simply takes the fixation task to the end
        """
        finish_fix_task = True
        while finish_fix_task:
            
            self.switch_fix_color()
            # if int(self.clock.getTime()*120) in self.fix_dot_color_timings:
            
            #     # change color
            #     self.fix_dot_color_idx += 1
            #     self.default_fix.setColor(self.fix_dot_colors[self.fix_dot_color_idx % len(self.fix_dot_colors)])

            self.default_fix.draw()
            if self.debug:
                self.debug_message.setText(f"ending fix, time: {self.clock.getTime(): .2f}, last one: {self.fix_dot_color_timings[-1]/120}\nblank time to do: {self.settings['stimuli']['blank_after_trs'] * self.TR}, topup time to do: {self.settings['mri']['topup_duration']}")
                self.debug_message.draw()

            self.win.flip()

            if int(self.clock.getTime()*120) > self.fix_dot_color_timings[-1]:
                finish_fix_task = False
        
        return

    def run(self):
        """ Runs experiment. """
        
        # self.display_text('', keys=self.settings['mri'].get('sync', 't'))
        # waitKeys(keyList = self.settings['mri'].get('sync', 't'))
        keys = []
        if self.debug:
            self.debug_message.setText("preparing to run, awaiting trigger")

        if self.eyetracker_on:
            self.calibrate_eyetracker()
            
            # make fix timings
            self.fix_dot_color_timings = self._make_fix_dot_color_timings(round_to = 2)
            if self.debug:
                print(f"created fix timings: {list(self.fix_dot_color_timings)}")
                # print(f"created fix timings (s): {[time/120 for time in self.fix_dot_color_timings]}")
            
            while self.settings['mri']['sync'] not in keys:
                keys = getKeys()
                # print(keys)

                # fix switch color needed
                self.switch_fix_color()

                if self.debug:
                    self.debug_message.draw()
                self.default_fix.draw()

                self.win.flip()

            self.start_experiment() # resets global clock as well
            self.start_recording_eyetracker()
        else:
            # make fix timings
            self.fix_dot_color_timings = self._make_fix_dot_color_timings(round_to = 2)
            
            if self.debug:
                print(f"created fix timings: {list(self.fix_dot_color_timings)}")
                # print(f"created fix timings (s): {[time/120 for time in self.fix_dot_color_timings]}")
            
            while self.settings['mri']['sync'] not in keys:
                keys = getKeys()
                # print(keys)

                # fix switch color needed
                self.switch_fix_color()

                if self.debug:
                    self.debug_message.draw()
                self.default_fix.draw()

                self.win.flip()

            self.start_experiment() # resets global clock as well

        for trial in self.trials:
            if self.debug:
                self.debug_message.setText("running trial {}".format(trial.trial_nr))
            
            trial.run()

        # finish task here, instead of dummy trial in the end
        self.end_experiment()

        self.close()

    def close(self):
        """'Closes' experiment. Should always be called, even when
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

        #save frame timings
        frametimings_df = pd.DataFrame(self.trialwise_frame_timings, columns = ["trial {}".format(str(i).zfill(2)) for i in range(self.n_trials)] )
        frametimings_df.to_csv(op.join(self.output_dir, self.output_str + "_frametimings.csv"), index = False)

        current_datetime = datetime.now()
        # plot and save audio
        if self.photodiode_check:
            photo_data = pd.DataFrame({'conditions':self.conditions,
                        'trial_type': self.trial_type,
                        'duration': self.recording_durations,
                        'delta_peaks': self.delta_peaks,
                        'n_peaks_found': self.n_peaks_found})
            
            photo_data.to_csv('photodiode_test_results/timing_photo_exp_results_{}.csv'.format(current_datetime.strftime("%Y-%m-%d-%H-%M")), index = False)

        # self.metadata['fix_dot_color_timings'] = list(self.fix_dot_color_timings)
        self.metadata['fix_dot_color_timings'] = [int(timing) for timing in self.fix_dot_color_timings]
        self.metadata['fix_dot_color_timings_effective'] = self.effective_fix_color_switches

        # save metadata
        json_path = os.path.join(self.output_dir, self.output_str + "_metadata.json")
        with open(json_path, "w") as json_file:
            json.dump(self.metadata, json_file, indent=4)

        if self.mri_simulator is not None:
            self.mri_simulator.stop()

        # calculate d'
        # according to https://wise.cgu.edu/wise-tutorials/tutorial-signal-detection-theory/signal-detection-d-defined-2/
        # confirmed with exercises
        n = len(self.effective_fix_color_switches)
        
        if self.n_fas >= n:
            # set arbitrary maximum fa_rate to allow d' calculation (also covering the unlikely edge case where hr > n)
            print(f"n_fas ({self.n_hits}) on {n} switches, setting fa_rate to (n-1)/n for d' calculation")
            fa_rate = (n-1)/n
        elif self.n_fas == 0:
            # set arbitrary minimum fa_rate to allow d' calculation
            print("no false alarms, setting fa_rate to 1/n for d' calculation")
            fa_rate = 1/n
        else:
            fa_rate = self.n_fas/n
        
        if self.n_hits >= n:
            # set arbitrary maximum hit_rate to allow d' calculation (also covering the unlikely edge case where hr > n)
            print(f"perfect hits ({self.n_hits}) on {n} switches, setting hit_rate to (n-1)/n for d' calculation")
            hit_rate = (n-1)/n
        elif self.n_hits == 0:
            # set arbitrary minimum hit_rate to allow d' calculation
            print(f"No hits ({self.n_hits}) on {n} switches, setting hit_rate to 1/n for d' calculation")
            hit_rate = 1/n 
        else:
            hit_rate = self.n_hits/n

        d_prime = norm.ppf(hit_rate) - norm.ppf(fa_rate)
        print(f'd_prime = {d_prime:.2f}, fa_rate = {fa_rate:.2f}, hit_rate = {hit_rate:.2f}')

        self.win.close()
        if self.eyetracker_on:
            self.stop_recording_eyetracker()
            self.tracker.setOfflineMode()
            core.wait(.5)
            f_out = op.join(self.output_dir, self.output_str + '.edf')
            self.tracker.receiveDataFile(self.edf_name, f_out)
            self.tracker.close()

        self.closed = True
