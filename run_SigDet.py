"""
- debug and check if all settings work out form command line
- investigate all outputs (regarding also things not checked so far, eg metacognition etc)
"""

from code.signal_detection_task import DetectSession
from datetime import datetime
import sys
import os


def main():
    sub = sys.argv[1] # which subject (XX)
    ses =  sys.argv[2] # which session (XX)
    run = sys.argv[3] # which run (XX)
    task = 'SigDet' # which task
    settings = f'settings/settings_{task}.yml' # grab settings
    dt = datetime.now().strftime('%Y%m%d%H%M%S') # get time to avoid overwriting

    # eyetracking yes/no
    eyetracker_on=True
    
    if 'tut' in sys.argv:
        run_tutorial = True
    else:
        run_tutorial = False
    if 'no-et' in sys.argv:
        eyetracker_on = False
        
    # generate output string
    output_str = f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_dt-{dt}'
    if 'test' in sys.argv:
       output_dir = f'data/TEST/psychophysics/sub-{sub}/ses-{ses}' # test for now
    else:
        output_dir = f'data/psychophysics/sub-{sub}/ses-{ses}'

    # Check if the directory already exists
    if not os.path.exists(output_dir):
        # Create the directory
        os.makedirs(output_dir)
        print("output_dir created successfully!")

    else:
        print("output_dir already exists!")

    print(f"running task {task} with {settings} for {output_str}, data saved in {output_dir}")
    print(f"eyetracking_on is {eyetracker_on} and run_tutorial is {run_tutorial}")

    # initialize TODO
    session = DetectSession(output_str, output_dir = output_dir, settings_file=settings,
                            eyetracker_on=eyetracker_on)
    
    # create trials
    session.create_yes_no_trials()
    # run the session
    session.run()

if __name__ == '__main__':
    main()