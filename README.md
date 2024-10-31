# Experiment code pertaining to sm-DN project

This repo contains task scripts and post-processing code pertaining to the sm-DN project.

An [exptools2](https://github.com/VU-Cog-Sci/exptools2/tree/master) installation is needed to run the tasks.

## Usage and Functionality
Tasks can be run from a terminal inside the root folder like:

```
cd sm-DN_tasks
python run_{task}.py {sub} {ses} {run}
```

For example, running signal detection for subject 03, session 02, run 01 requires:

```
python run_SigDet.py 03 02 01
```

For testing, this call can be either supplemented with an additional argument `test` or one of `{sub} {ses} {run}` replaced with `test`. This will place the resulting data in a `TEST` sub-directory.

The root-level scripts prepended with `run` use task code and assets in the `code` subdirectory and settings in the `settings` subdirectory.

## Outputs
The task outputs are prepended with a [BIDS](https://bids-specification.readthedocs.io/en/stable/)-like string: `f'sub-{sub}_ses-{ses}_task-{task}_run-{run}''` and are additionally datetime-stamped with `f'dt-{datetime.now().strftime('%Y%m%d%H%M%S')}'`. Outputs for all experiments include:
1. `'_events.tsv'`: The task events log
2. `'_expsettings.yml'`: A mirror of the used settings
3. `'_log.txt'`: The psychopy log
4. `'_frames.pdf'`: A pdf containing a visualization of frame timings across the experiment

In addition, both the CTS and HRF tasks output:
1. `'_metadata.json'`: Some metadata relating to the experiment

In addition, the CTS task outputs:
1. `'_frametimings.csv'`: a file tracking frame timings within stimulus presentation

## Psychophysics tasks
### Temporal Integration: temporal order judgement (TempIntTOJ)
The participant is presented with both an auditory and a visual stimulus in two conditions (audio first or visual first) across different stimulus onset asynchronies (phase 2). After, the participant needs to indicate which one was first while the fixation is green, indicating the response period (phase 3). If no response was given, the fixation dot turns black shortly (phase 4). After, the dot turns blue indicating the confidence response period, where the participant responds with either high or low confidence (phase 5). Again, if no response was given, the dot turns black for feedback (phase 6).

### Temporal Integration: synchrony judgement (TempIntSJ)
The participant is presented with both an auditory and visual stimulus in two conditions (audio first or visual first) across different stimulus onset asynchronies (phase 2). After, the participant needs to indicate whether they perceived the stimulus as synchonous or asynchonous while the fixation is green, indicating the response period (phase 3). If no response was given, the fixation dot turns black shortly (phase 4). After, the dot turns blue indicating the confidence response period, where the participant responds with either high or low confidence (phase 5). Again, if no response was given, the dot turns black for feedback (phase 6).

TODO edit
### Temporal Reproduction (TempRep)
The participant is presented with a visual stimulus for a certian duration (phase 2). After a short interval (phase 3, 500 ms), the dot turns green and the participant needs to reproduce that duration by holding down the right button on the box or the spacebar (phase 4). If no response was given, a the fixation dot turns black shortly (phase 5).

### Signal Detection (SigDet)
The participant is presented with moving noise, in which a Gabor grating may appear (phase 2, on 50% of trials). The fixation will turn green to indicate the response period in which the participant will respond whether they saw a pattern or not (phase 3). If no answer is given, the dot will turn black shortly (phase 4). After, the dot turns blue indicating the confidence response period, where the participant responds with either high or low confidence (phase 5). Again, if no response was given, the dot turns black for feedback (phase 6).

#### Notable settings

```yml
response:
  device: 'button_box' # 'keyboard' or 'button_box', defines mapping of buttons in session.__init__
  metacognition: 'split' # 'split' or 'together', detection and metacog used to be taken in one button press ('together')
stimuli:
  n_noise_textures: 25 # amount of noise textures to use
  signal_parameters:
    spatial_freq: 5 # in dva
    size: 1024 # in pixels
    signals: [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05] # signals in psychopy opacity values
  noise_parameters:
    refresh_frame: 8 # on which modulo of number frames screen should be refreshed
```

## MRI tasks
TODO edit
