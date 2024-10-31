import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf

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
