import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf
import itertools

# define comulative normal
def cumulative_normal(x, alpha, beta):
    # Cumulative distribution function for the standard normal distribution
    return 0.5 + 0.5 * erf((x - alpha) / (beta * np.sqrt(2)))

def cumulative_normal_lapse(x, alpha, beta, lamb):
    # Cumulative distribution function for the standard normal distribution
    return lamb + (1-2 * lamb) * (0.5 + 0.5 * erf((x - alpha) / (beta * np.sqrt(2))))

# Define Gaussian function
def gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))

def quick_fit(x_data, y_data, model, init, ax=None, plot = False, additional_points = None, x_range_fit=None):
    """
    Performs a quick fit of the y_data at x_data coordinates to the given model with starting
    parameters init. Optionally plots the data, fit and additional points on the fit.
    
    init is a dict with {param_name : value} that matches the parameters expected by the model
    """
    
    # get initial guess and param names
    initial_guess = list(init.values())
    param_names = list(init.keys())
    
    # Fit the data
    params, _ = curve_fit(model, x_data, y_data, p0=initial_guess)
    
    if plot:
        # infer x for plotting fit
        if x_range_fit is None:
            x_plot = np.linspace(x_data.min(), x_data.max(), 100)
        else:
            x_plot = np.linspace(x_range_fit[0], x_range_fit[1], 100)
            
        # generate text with parameter names and values
        fitted_text = '\n'.join([f'{name} = {param:.4f}' for name, param in zip(param_names, params)])

        if not ax:
            # make ax if ax is not given
            fig, ax = plt.subplots()
            
        # plot data and fit
        ax.scatter(x_data, y_data, label='Data', color='red')
        ax.plot(x_plot, model(x_plot, *params), label='Fitted Gaussian', color='blue')
        
        if additional_points is not None:
            ax.scatter(additional_points, model(additional_points, *params), color = 'green')#  facecolors='none', edgecolors='green')
        
        ax.text(0.05, 0.95, fitted_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top')
    
    if plot:
        return params, fig, ax
    else:
        return params

def select_data_tempInt(subnrs, sesnrs, runnrs, root = 'TempInt_pilot', task = 'all', verbose = 1):
    """
    function going over folders to find all filepaths of the given sub-ses-run combination
    """
    paths = []
    # going through all possible sub/ses/run combinations 
    for subnr, sesnr, runnr in itertools.product(subnrs, sesnrs, runnrs):
        # get paths to events.tsv and expsettings.yml
        data_path = f'{root}/sub-{subnr}/ses-{sesnr}/sub-{subnr}_ses-{sesnr}_task-TempInt_run-{runnr}_events.tsv'
        settings_path = f'{root}/sub-{subnr}/ses-{sesnr}/sub-{subnr}_ses-{sesnr}_task-TempInt_run-{runnr}_expsettings.yml'
        
        try:
            # check if sub/ses/run combination exists
            dat = pd.read_csv(data_path, sep='\t')
            if verbose:
                print(f"Found {data_path}")

        except:
            # if not, continue with next
            if verbose:
                print(f"Not found {data_path}")
            continue
        
        # load settings file
        with open(settings_path, 'r') as file:
            settings = yaml.safe_load(file)
    
        # check if this is indeed the expected task (TOJ/SJ)
        if task != 'all':
            if settings['task']['type']==task:
                # save this path
                paths.append(data_path)

            else:
                if verbose: 
                    print(f"given task {task} is not congruent with info in settings file: {settings['task']['type']}")
                    print(f"check if the given run/task combination is correct")
                else:
                    continue
        else:
            paths.append(data_path)

    return paths

# subnrs = ['01', '06', '07', '08', '09']
# sesnrs = ['01', '02']
# runnrs = ['01', '02']

# select_data_tempInt(subnrs, sesnrs, runnrs, task = 'TOJ', verbose = 0)

def parse_bids_filename(path):
    """
    splits a bids filename into a dict
    """
    bids_dict = {}
    
    split = path.split('/')[-1].split('_')#[0].split('-')[-1]
    # make dict
    for part in split:
        if '-' in part:
            bids_dict[part.split('-')[0]] = part.split('-')[1] 
    return bids_dict


def prep_data_tempInt(subnrs, sesnrs, runnrs, task='SJ', root = 'TempInt_pilot', framerate = 1/120, verbose = 0,
                     return_paths = False):
    """
    goes through root folder, should be bids-like, selects all data for the
    given sub-ses-run-task combination
    
    one just processing the events.tsv and the other one managing the files,
    add sigmoid and gaussian fits
    """
    # init result df lists, to become columns
    soas = [] # stimulus onset asynchrony
    resps = [] # responses
    subs = [] # subject
    sessions = []
    runs = []
    which_first = [] # A or V first?
    
    # select data
    data_paths = select_data_tempInt(subnrs, sesnrs, runnrs, root = root, task = task, verbose = 0)

    for path in data_paths:
        # load run data
        dat = pd.read_csv(path, sep='\t')
        
        # unpack sub, ses, run from string
        bids_dict = parse_bids_filename(path)
        subnr = bids_dict['sub']
        sesnr = bids_dict['sess']
        runnr = bids_dict['run']
                    
        for trial in dat.trial_nr.unique():    
            # index trial
            trial_df = dat[dat.trial_nr == trial]
            soa = trial_df.soa.iloc[0]

            if task == 'SJ':
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
            subs.append(subnr)
            sessions.append(sesnr)
            runs.append(runnr)
            which_first.append(first)
    
    # make df
    resps_df = pd.DataFrame({'soa_f' : soas, 'neg_soa_f': [-soa for soa in soas], 'soa_ms': [soa*framerate*1000 for soa in soas], 
                             'neg_soa_ms': [-soa*framerate*1000 for soa in soas],
                             'response' : resps, 'subject': subs, 'session':sessions, 'run':runs, 'first':which_first})

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
    
    if return_paths:
        return resps_df, data_paths
    else:
        return resps_df

    