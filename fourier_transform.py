"""Documentation complete"""

#data import
import pandas as pd

#data visualization
import matplotlib.pyplot as plt

#logging functions
from Code_Logger import log_functions as lg

#calculation modules
import numpy as np
from custom_math import gauss_optimum
from custom_math import find_extrema



def DFT(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x)
    N = x.shape[0]
    j = np.arange(N)
    k = j.reshape((N, 1))
    W = np.exp(-2j * np.pi * j * k / N)
    lg.text_log('DFT calculation via matrix multiplication')
    return np.dot(W, x)

def iDFT(x_d): 
    """Compute the inverse discrete Fourier Transform of the 1D array x"""
    x_d = np.asarray(x_d)
    N = x_d.shape[0]
    j = np.arange(N)
    k = j.reshape((N, 1))
    W_1 = np.exp(2j * np.pi * j * k / N)
    lg.text_log('iDFT calculation via matrix multiplication')
    return 1/N*np.dot(W_1, x_d)  
    
def fourier_transform(dt, I):
    """
    Calculates discrete or fast fourier transform. Depends on signal array size

    Parameters
    ----------
    dt : float
        signal time point distance.
    I : list or array
        signal intensity array.

    Returns
    -------
    f_range : numpy array
        frequency range.
    spectrum : numpy array
        complex fourier coefficients.

    """
    exp = np.log(len(I))/np.log(2)
    lg.text_log('compute discrete or fast fourier transform, depending on signal array size')
    if exp.is_integer():
        lg.text_log('numpy fast fourier transform (CT Algorithm)')
        F_I = np.fft.fft(I)
    else:
        F_I = DFT(I)
    
    lg.text_log('calculate frequency increment: f_inc = 1/(n_f*2*dt) with dt: time delta and n_f number of timepoints/2')
    n_f = int(len(F_I)/2)
    f_inc = 1/(n_f*2*dt)
    
    lg.text_log('frequency spectrum without aliased frequencies is returned')
    spectrum = F_I[1:(n_f)]
    f_range = np.arange(1*f_inc,f_inc*n_f, f_inc)
    
    return f_range, spectrum
  
def fourier_params(dt, I, frequency = None, plot = None):
   
    #calculate fourier transform
    f_range, spectrum = fourier_transform(dt, I)
    
    #get values from fourier transform
    lg.text_log('get values from fourier transform')
    magnitude_list = np.multiply(np.abs(spectrum),(1/(len(spectrum)+1)))
    phase_list = np.angle(spectrum)
    real_list = np.real(spectrum)
    imag_list = np.imag(spectrum)
    
    #optional plots
    if plot == 'magnitude':
        plt.plot(f_range, magnitude_list)
        
    if plot == 'phase':
        plt.plot(f_range, phase_list)
        
    if plot == 'real':
        plt.plot(f_range, real_list)
        
    if plot == 'imag':
        plt.plot(f_range, imag_list)
    
    #get index of highest magnitude value
    
    lg.text_log('get parameters for defined frequency or frequency with highest magnitude')
    if frequency == None:
        mag_sort_indices = np.argsort(magnitude_list)
        f_index = mag_sort_indices[-1]
    else:
        diff_f = [abs(f_i-frequency) for f_i in f_range]
        f_index = [list(diff_f).index(x) for x in diff_f if x == min(diff_f)][0]

    
    #get corresponding values for highest magnitude
    frequency = f_range[f_index]
    magnitude = magnitude_list[f_index]
    phase = phase_list[f_index]
    real = real_list[f_index]
    imag = imag_list[f_index]
    
    return frequency, magnitude, phase, real, imag

def low_pass_gauss_filter(n, sigma_filter):
    lg.function_log()
    
    lg.text_log('filtering high frequencies - defined by sigma')
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
    lg.text_log('calculate fourier transform')
    f_range, spectrum = fourier_transform(dt, I)
    lg.text_log('get magnitude from spectrum')
    magnitude = gauss_optimum(np.abs(spectrum))
    
    
    lg.text_log('get extrema of fourier transform magnitude')
    max_x, max_y, min_x, min_y = lg.var_log(find_extrema(f_range, magnitude))
    
    lg.text_log('get cutoff frequency - at first minimum')
    bg_cutoff = lg.var_log(min((min_x[np.argsort(min_y)[-1]]), (max_x[np.argsort(max_y)[-1]])))
    
    lg.text_log('get indices below cutoff')
    bg_indices = lg.var_log([list(f_range).index(x) for x in f_range if x < bg_cutoff])
    
    lg.text_log('maximum index*0.5 yields sigma for gauss filtering ')
    sigma_filter = lg.var_log(max(bg_indices)/2)
    
    #create full spectrum for filtering and inverse fourier transformation
    full_spectrum = DFT(I)
    #create gauss filter
    gauss_filter = np.asarray(low_pass_gauss_filter(len(full_spectrum), sigma_filter)) 
    #filter spectrum by multiplication with gauss filter
    filtered_spectrum = np.multiply(full_spectrum, gauss_filter)
    
    normalized_magnitude = np.multiply(np.abs(full_spectrum[1:]), 1/max(np.abs(full_spectrum[1:])))
    
    lg.img_log((normalized_magnitude, gauss_filter[1:]), title='gauss filter',
               plot='yn', legend=['raw spectrum','gauss_fiter'],
               x_axis = 'n', y_axis = 'magnitude')
    
    #create baseline via inverse fourier transformation
    baseline = np.abs(iDFT(np.asarray(filtered_spectrum)))
    

    t = np.arange(0,len(I)*dt,dt)
    coef = np.polyfit(t,baseline,1)
    inclination = coef[0]
    poly1d_fn = np.poly1d(coef)

    lg.img_log((t, (I, baseline, poly1d_fn(t))), title = 'baseline calculation',
               legend=['raw signal', 'fourier filtering baseline', 'line fit'],
               x_axis='t [s]', y_axis='I A.U.')  
    
    return baseline, inclination    

def demo_fourier_transform():
    
    lg.log_init()
    file_1 = r'E:\PhD\Finale Übergabe\EIM elektrisch\Methode\246_EIM elektrisch\fft test.txt'
    data = pd.read_csv(file_1, sep='\t', encoding='cp1252')
    data.columns = ["t", "I"]
    I_1 = data["I"]
    dt_1 = data["I"][1]-data["I"][0]
    
    file_2 = r'E:\PhD\Finale Übergabe\EIM elektrisch\Methode\246_EIM elektrisch\fft test 2.txt'
    data = pd.read_csv(file_2, sep='\t', encoding='cp1252')
    data.columns = ["t", "I"]
    I_2 = data["I"]
    dt_2 = data["I"][1]-data["I"][0]
    
    f_range_1, spectrum_1 = fourier_transform(dt_1, I_1)
    f_range_2, spectrum_2 = fourier_transform(dt_2, I_2)
    
    
    lg.img_log(((f_range_1, np.abs(spectrum_1)),(f_range_2, np.abs(spectrum_2))), title = 'fourier transform',
                            legend = ['file 1, n = '+str(len(I_1)), 'file 2, n = '+str(len(I_2))],
                            x_axis='frequency [Hz]', y_axis= 'Magnitude A.U.')

    #test fourier transform back and forth
    """
    DFT_calc = np.abs(iDFT(DFT(I_2)))
    fft_calc = np.abs(np.fft.ifft(np.fft.fft(I_1))).transpose()
    plt.plot(fft_calc, label = 'iDFT(DFT(I_1))')
    plt.plot(DFT_calc, label = 'iDFT(DFT(I_2))')
    plt.legend()
    plt.show()
    """

def demo_baseline():
    
    lg.log_init()
    #file_path = 'fft test.txt'
    file_path = 'baseline_sample.txt'
    data = pd.read_csv(file_path, sep='\t', encoding='cp1252')
    data.columns = ["t", "I"] 
    
    dt = 0.1
    I = data['I']
    base, inclination = baseline(dt, I)

#demo_baseline()