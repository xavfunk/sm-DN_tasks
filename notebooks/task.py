import itertools
import os

from linescanning import dataset

class Subject():
    def __init__(self, subnr):
        self.subnr = subnr

class Task():
    """
    parent class for all tasks
    """

    def __init__(self, subject, root, verbose = False):
        """
        subject is an instance of the Subject class
        root is the path to the root of the data folder (where /sub-XX folders)
        are located
        """
        
        self.subject = subject
        self.subnr = subject.subnr
        self.task_name = None
        self.root = root
        self.verbose = verbose

    def grab_files(self, sessions, runs, combine_runs=False):
        """
        constructs all possible paths (sessions x runs) to events.tsv, .edf and 
        expsettings.yml files and verifies them for existence.
        sessions, runs are iterables of which sessions and runs to grab
        """

        # raise exception if we are not in a subclass
        if self.task_name is None:
            raise NotImplementedError(f"Task name is None, but needs to be \
                                      defined in order to grab files")
        
        # path templates for sub/ses/task/run combination
        path_templates = [os.path.join(self.root, f'sub-{self.subnr}', f'ses-{ses}', \
                        f"sub-{self.subnr}_ses-{ses}_task-{self.task_name}_run-{run}") \
                        for ses, run in itertools.product(sessions, runs)]
        
        # construct events paths
        self.events_paths = [path + '_events.tsv' for path in path_templates]
        # filter for existence
        self.events_paths = [path for path in self.events_paths if os.path.exists(path)]

        # construct eyetracking paths
        self.et_paths = [path + '.edf' for path in path_templates]
        # filter for existence
        self.et_paths = [path for path in self.et_paths if os.path.exists(path)]

        # construct settings paths
        self.expsettings_paths = [path + '_expsettings.yml' for path in path_templates]
        # filter for existence
        self.expsettings_paths = [path for path in self.et_paths if os.path.exists(path)]

        if self.verbose:
            print(f'Found the following files for subject {self.subnr}, sessios {sessions}, runs {runs}')
            print('events.tsv:')
            for ev in self.events_paths:
                print(ev)

            print('.edf:')
            for et in self.et_paths:
                print(et)
    
            print('_expsettings.yml:')
            for set in self.expsettings_paths:
                print(set)

    def load_settings():
        """
        TODO
        simply loads the settings file
        """

        pass

    def prep_eyetracking(self):
        
        # get all found ses, run combinations
        for path in self.et_paths:
            filename, _ = os.path.splitext(path.split(os.path.sep)[-1])
            ses = filename.split('_')[1].split('-')[1]
            run = filename.split('_')[-1].split('-')[1]
            print(f'found et data for {ses}, {run}')
            
            # keep track of found ses, run combinations

        # load eyetracking files
        
        
        pass

    def plot_eyetracking():
        pass

    def fit_data():
        pass

    def plot_fits():
        pass


class TempRep(Task):
    def __init__(self, subject, root, verbose = False):
        super().__init__(subject, root, verbose = verbose)
        self.task_name = 'TempRep'
        self.grab_files(['00', '01', '02', '03'], ['01'])

    def prep_data():
        pass

    def fit_data():
        pass

    def plot_fits():
        pass
        
    def plot_data():
        pass


class SigDet(Task):
    def __init__(self, subject):
        super().__init__(subject)
        self.task_name = 'SigDet'

class TempIntTOJ(Task):
    def __init__(self, subject):
        super().__init__(subject)
        self.task_name = 'TempIntTOJ'

class TempIntSJ(Task):
    def __init__(self, subject):
        super().__init__(subject)
        self.task_name = 'TempIntSJ'

class CTS(Task):
    def __init__(self, subject):
        super().__init__(subject)
        self.task_name = 'CTS'

class HRF(Task):
    def __init__(self, subject):
        super().__init__(subject)
        self.task_name = 'HRF'

if __name__ == '__main__':

    ## Testing Code
    root = '/home/funk/repos/sm-DN_tasks/data/TEST/psychophysics'
    sub = Subject('01')
    task = TempRep(sub, root, verbose = True)
    task.prep_eyetracking()
    # print(sub.__dict__)
    # print(task.__dict__)

    pass