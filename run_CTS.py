"""
- debug and check if all settings work out form command line
- investigate all outputs (regarding also things not checked so far, eg metacognition etc)
"""

from code.CTS_sess import DelayedNormSession
from datetime import datetime
import sys
import os


def main():
    sub = sys.argv[1] # which subject (XX)
    ses =  sys.argv[2] # which session (XX)
    run = sys.argv[3] # which run (XX)
    task = 'CTS' # which task
    settings = f'settings/settings_{task}.yml' # grab settings
    dt = datetime.now().strftime('%Y%m%d%H%M%S') # get time to avoid overwriting

    # eyetracking yes/no
    eyetracker_on=True
    if 'no-et' in sys.argv:
        eyetracker_on = False

    # debug yes/no
    debug=False
    if 'debug' in sys.argv:
        debug = True

    # generate output string
    output_str = f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_dt-{dt}'
    if 'test' in sys.argv:
       output_dir = f'data/TEST/mri/sub-{sub}/ses-{ses}' # test for now
    else:
        output_dir = f'data/mri/sub-{sub}/ses-{ses}'

    # Check if the directory already exists
    if not os.path.exists(output_dir):
        # Create the directory
        os.makedirs(output_dir)
        print("output_dir created successfully!")

    else:
        print("output_dir already exists!")

    print(f"running task {task} with {settings} for {output_str}, data saved in {output_dir}")
    print(f"eyetracking_on is {eyetracker_on}")

    # initialize
    session = DelayedNormSession(output_str, output_dir = output_dir, eyetracker_on=eyetracker_on,
                            n_trials=None, settings_file=settings, photodiode_check = False, debug = debug)
    # create trials
    session.create_trials()
    # run the session
    session.run()
    # TODO is necessary?
    session.quit()

if __name__ == '__main__':
    main()