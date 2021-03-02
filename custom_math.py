

#calculation modules
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from helper_functions import fit_params

#logging functions
import log_functions as lg


def gauss_peak(x, Amp, sigma, cen):
    return Amp*np.exp(-((x-cen)**2)*0.5/sigma**2)

def find_extrema(x, y):
    lg.function_log()
    
    lg.text_log('fit spline for algebraic operations')
    y_spline = UnivariateSpline(x, y, k=4, s=0)
    
    lg.text_log('get extrema via roots of first derivative')
    d_roots = y_spline.derivative().roots()
    
    
    lg.img_log((x,(curve_normalize(y),curve_normalize(y_spline(x)),curve_normalize(y_spline.derivative()(x)))),
               title='extrema calculation (normalized)', legend=['y','spline fit','1st derivative (spline)'],
               x_axis='x', y_axis='y')
    
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
        
    lg.text_log('set initial d_args peak parameters')
    Amp_init = d_args_list[np.argsort(d_args_list)[-1]]
    cen_init = sigma_range[np.argsort(d_args_list)[-1]]
    sigma_init = abs(cen_init-sigma_range[0])
    
    p_Amp = [Amp_init, 0, Amp_init*1.2]
    p_cen = [cen_init, 0, cen_init*2]
    p_sigma = [sigma_init, sigma_init*0.01, sigma_init*5]
    
    params, bnds = lg.var_log(fit_params((p_Amp, p_sigma, p_cen)))
    
    lg.text_log('fit gauss peak to extract optimum sigma')  
    gauss_peak_params, gauss_peak_errs = lg.var_log(curve_fit(gauss_peak, sigma_range, d_args_list, p0=params, bounds=bnds))
    lg.var_log(gauss_peak_params)

    #documentation plot
    lg.img_log((sigma_range, (d_args_list, gauss_peak(sigma_range, *gauss_peak_params))),
               title='argsort difference peak fit', legend=['d_args raw', 'gauss peak fit'],
               x_axis='sigma', y_axis='d_args')
             
    
    d_args_list = gauss_peak(sigma_range, *gauss_peak_params)           
        
    lg.text_log('get optimum sigma for gauss filtering from maximum argsort difference')
    sigma_optimum = lg.var_log(gauss_peak_params[2]+abs(gauss_peak_params[1]))
    lg.text_log('calculated new filtered spectrum')
    I_optimum = gaussian_filter1d(I, sigma_optimum)
    
    
    #documentation plot
    lg.img_log((I, I_optimum),
           title='gauss_filtering', plot='yn', legend=['original', 'sigma optimum'],
           x_axis='n', y_axis='magnitude')
    
    return I_optimum 

def curve_normalize(data):
    result = []
        
    for i in data:
        result.append(i/max(abs(data)))
        
    return result
    