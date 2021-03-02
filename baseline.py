
#data import
import pandas as pd

#data visualization
import matplotlib.pyplot as plt

#calculation modules
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

#custom fourier transform modules
from fourier_transform import fourier_transform
from fourier_transform import iDFT
from fourier_transform import DFT

import log_functions as lg
import inspect

from helper_functions import fit_params

def find_extrema(x, y):
    lg.function_log()
    
    lg.text_log('fit spline for algebraic operations')
    y_spline = UnivariateSpline(x, y, k=4, s=0)
    """
    plt.plot(x, y, label = 'original')
    plt.plot(x, y_spline(x), label = 'spline')
    #plt.plot(x, y_spline.derivative()(x), label = '1st derivative')
    plt.xlabel('frequency [Hz]')
    plt.ylabel('magnitude')
    lg.img_log('find extrema')
    """
    
    lg.text_log('get extrema via roots of first derivative')
    d_roots = y_spline.derivative().roots()
    
    lg.text_log('assign extrema via second derivative')
    max_x = [x for x in d_roots if y_spline.derivative(2)(x) < 0]
    min_x = [x for x in d_roots if y_spline.derivative(2)(x) > 0]
        
    return max_x, y_spline(max_x), min_x, y_spline(min_x)    

def gauss_optimum(I):
    lg.function_log()
    
    cost_list = []
    dI_list = []
    d_args_list = []
    
    sigma_factor = 0.1

    sigma_range = np.arange(1*sigma_factor,len(I)*sigma_factor,sigma_factor)
    I_0 = I
    
    lg.text_log('calculate squared argsort difference for I_n-1 and I_n ')
    
    for sigma in sigma_range:
        
        I_gauss = gaussian_filter1d(I, sigma)
        
        dI = sum([(a_i - b_i)**2 for a_i, b_i in zip(I_gauss, I_0)])
        
        d_args = sum([(a_i - b_i)**2 for a_i, b_i in zip(np.argsort(I_gauss), np.argsort(I_0))])
        
        cost = d_args*dI
        
        I_0 = I_gauss
        
        cost_list.append(cost)
        dI_list.append(dI)
        d_args_list.append(d_args)
        
    #plt.plot(sigma_range, d_args_list, label = 'd_args')
    #lg.img_log('argsort difference raw')
    
    #lg.text_log('set initial d_args peak parameters')
    Amp_init = d_args_list[np.argsort(d_args_list)[-1]]
    cen_init = sigma_range[np.argsort(d_args_list)[-1]]
    sigma_init = abs(cen_init-sigma_range[0])
    
    p_Amp = [Amp_init, 0, Amp_init*1.2]
    p_cen = [cen_init, 0, cen_init*2]
    p_sigma = [sigma_init, sigma_init*0.01, sigma_init*5]
    
    params, bnds = lg.var_log(fit_params((p_Amp, p_sigma, p_cen)))
    
    
    #lg.text_log('fit gauss peak to extract optimum sigma')  
    gauss_peak_params, gauss_peak_errs = curve_fit(gauss_peak, sigma_range, d_args_list, p0=params, bounds=bnds)
    #lg.var_log(gauss_peak_params)
    
    
    #plt.plot(sigma_range, d_args_list, label = 'd_args')
    d_args_list = gauss_peak(sigma_range, *gauss_peak_params)
    #plt.plot(sigma_range, d_args_list, label = 'gauss peak fit')
    #plt.xlabel('sigma')
    #plt.ylabel('d_args')
    #lg.img_log('argsort difference peak fit')
    
    #lg.text_log('get optimum sigma for gauss filtering from maximum argsort difference')
    sigma_optimum = lg.var_log(gauss_peak_params[2]+abs(gauss_peak_params[1]))
    #lg.text_log('calculated new filtered spectrum')
    I_optimum = gaussian_filter1d(I, sigma_optimum)
    
    #documentation plot
    #plt.plot(I, label='original')
    #plt.plot(I_optimum, label='sigma optimum')
    #plt.xlabel('n')
    #plt.ylabel('magnitude')
    #lg.img_log('gauss_filtering')
    
    return I_optimum    
        
def filter_plot(x,y, title = 'test plot'):
    #plot function comparing the filter against the optimum filter result
    plt.plot(x,y, label = 'original')
    plt.plot(x, gauss_optimum(y), label = 'optimal filter')
    plt.legend()
    plt.show()
    
def gauss_peak(x, Amp, sigma, cen):
    return Amp*np.exp(-((x-cen)**2)*0.5/sigma**2)

def low_pass_gauss_filter(n, sigma_filter):
    lg.function_log()
    
    #lg.text_log('filtering high frequencies - defined by sigma')
    x = np.arange(0,n,1)
    
    
    return (np.exp(-(x**2)*0.5/sigma_filter**2)+np.exp(-(((x-n+1)**2)*0.5/sigma_filter**2)))
    
def baseline(dt, I, report = True, logfile = None):
    """

    Parameters
    ----------
    dt : float;
    distance between time points.
    I : array;
    Signal Intensity/Magnitude.

    Returns
    -------
    baseline : List;
    Intensity array for baseline subtraction.
    inclination : float;
    inclination of baseline calculated via linear regression.

    """
    
    lg.function_log()
    #create spectrum to analyze
    #lg.text_log('calculate fourier transform')
    f_range, spectrum = fourier_transform(dt, I)
    #lg.text_log('get magnitude from spectrum')
    magnitude = gauss_optimum(np.abs(spectrum))
    
    
    #lg.text_log('get extrema of fourier transform magnitude')
    max_x, max_y, min_x, min_y = lg.var_log(find_extrema(f_range, magnitude))
    #print(find_extrema(f_range, magnitude))
    
    #lg.text_log('get cutoff frequency - at first minimum')
    bg_cutoff = lg.var_log(min((min_x[np.argsort(min_y)[-1]]), (max_x[np.argsort(max_y)[-1]])))
    
    #lg.text_log('get indices below cutoff')
    bg_indices = lg.var_log([list(f_range).index(x) for x in f_range if x < bg_cutoff])
    
    #lg.text_log('maximum index*0.5 yields sigma for gauss filtering ')
    sigma_filter = lg.var_log(max(bg_indices)/2)
    
    
    #create full spectrum for filtering and inverse fourier transformation
    full_spectrum = DFT(I)
    #create gauss filter
    gauss_filter = np.asarray(low_pass_gauss_filter(len(full_spectrum), sigma_filter)) 
    #filter spectrum by multiplication with gauss filter
    filtered_spectrum = np.multiply(full_spectrum, gauss_filter)
    #create baseline via inverse fourier transformation
    baseline = np.abs(iDFT(np.asarray(filtered_spectrum)))
    

    t = np.arange(0,len(I)*dt,dt)
    coef = np.polyfit(t,baseline,1)
    inclination = coef[0]
    
    poly1d_fn = np.poly1d(coef)
    #plt.plot(t,baseline, label = 'fourier filtering baseline')
    #plt.plot(t, poly1d_fn(t), label = 'line fit')
    #plt.xlabel('t [s]')
    #plt.ylabel('I A.U.')
    #lg.img_log('baseline')
    
    return baseline, inclination
        
def demo():

    file_path = r'E:\PhD\Finale Übergabe\EIM elektrisch\Methode\246_EIM elektrisch\fft test.txt'
    #file_path = r'E:\PhD\Finale Übergabe\EIM elektrisch\Methode\246_EIM elektrisch\baseline_sample.txt'
    data = pd.read_csv(file_path, sep='\t', encoding='cp1252')
    data.columns = ["t", "I"] 
    
    dt = 0.1
    t = np.multiply(data['t'], dt)
    
    I = data['I']
    filter_plot(t,I, title = 'signal')

    f_range, spectrum = fourier_transform(dt, gauss_optimum(I))
    magnitude = np.abs(spectrum)
    
    filter_plot(f_range, magnitude, title = 'spectrum')
    
    #I_gauss = gauss_optimum(I)
    
    base, inclination = baseline(dt, I)
    
    final_signal = I-base
    plt.plot(t, final_signal)
    plt.show()
       

    
    













