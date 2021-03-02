"""Documentation complete"""

#image import
from skimage import io

#image export
from PIL import Image

#calculation modules
import numpy as np

#custom fourier transform modules
from fourier_transform import fourier_params
from fourier_transform import fourier_transform
from fourier_transform import baseline

#calculation duration estimation
from datetime import datetime
from datetime import timedelta

import log_functions as lg

def image_fft(window_size, file_path, dt, frequency = None):
    
    #load image
    lg.text_log('loading raw image stack')
    im = io.imread(file_path)
    
    #correction in case window size is set larger the image size
    window_size = min(window_size, im.shape[1]-1, im.shape[2]-1)
    
    #get image dimensions
    x_dim = (im.shape[1]-window_size)
    y_dim = (im.shape[2]-window_size)
    z_dim = im.shape[0]
    
    #create arrays to iterate over
    x_range = np.arange(0,x_dim)
    #x_range = np.arange(0,2)
    y_range = np.arange(0,y_dim)   
    z_range = np.arange(0,z_dim)
    
    #create empty result matrix to store the calculated values
    result = np.empty((6, x_dim, y_dim))
    
    #total dimensions for measurement of calculation progress
    n_total = x_dim*y_dim*z_dim
    
    #count for estimation of calculation progress
    count = 0
    
    #iteration over all pixels (x,y) and slides z
    print('start calculation...')
    
    start_time = datetime.now()
    elapsed_time = datetime.now()-start_time

    t_log = open(file_path[:-4]+'_progress_log.txt', "w")
    t_log.write('total dimensions: '+str(n_total)+'\n')

    
    lg.text_log('calculate fourier transform for each pixel')
    lg.text_log('the mean intensity for each pixel with a defined window size (x^2) is calculated')
    for x in x_range:
    
        if count == 0:
            remaining_time = ''
        else:
            remaining_time = str(timedelta(seconds=(elapsed_time.seconds/count*(n_total-count))))
            
        progress = np.round((count/n_total*100),1)
        print(str(progress)+' % done. remaining: '+remaining_time)
        t_log.write(str(elapsed_time.seconds)+'\t'+remaining_time+'\t'+str(elapsed_time.seconds/max(count,1))+'\n')
    
        for y in y_range:
            
            #signal array to store the mean value of the window
            signal_array = []
            
            #print progress

 
            for z in z_range:

                #collecting all pixel values in the window
                img_slice = im[z]
                rows = img_slice[x:x+window_size]
                full_array = []
                for item in rows:
                    full_array = full_array + list(item[y:y+window_size])
                
                #append mean values to signal array
                signal_array.append(np.mean(full_array))
                #count for estimation of calculation progress
                count = count+1
            
            elapsed_time = datetime.now()-start_time
            #baseline and its inclination
            lg.text_log('optional baseline calculation and correction')
            
            """
            base, inclination = baseline(dt, signal_array)
            
            
            lg.img_log((fourier_transform(dt, signal_array)[0],
                    (np.abs(fourier_transform(dt, signal_array)[1]), np.abs(fourier_transform(dt, signal_array-base)[1]))),
                    title='baseline subtraction impact', legend=['raw','baseline subtracted'],
                    x_axis='frequency [Hz]', y_axis='Magnitude A.U.')
            
            signal_array = signal_array-base         
            """
            #calculate fourier parameters of baseline corrected signal
            lg.text_log('get complex impedance parameters from fourier transform:')
            freq, mag, phase, real, imag = fourier_params(dt, signal_array, frequency)
            
            #write results to corresponding image slides
            result[0,x,y]= freq
            result[1,x,y]= mag
            result[2,x,y]= phase
            result[3,x,y] = real
            result[4,x,y] = imag
            #result[5,x,y]= inclination
    t_log.close()
    
    return result         

def save_image(data, file_path):
    lg.text_log('saving images as tif files')
    lg.text_log('each image is a complex impedance parameter: frequency, magnitude, phase, Re, Im and optional baseline slope')
    file_path_list = file_path.replace('\\','/').split('/')
    file_path = '/'.join(map(str, file_path_list[:-1]))
    file_name = file_path_list[-1].split('.')[0] 
    
    img = Image.fromarray(data[0])
    img.save(file_path+'/'+file_name+'_freq.tif')
    
    img = Image.fromarray(data[1])
    img.save(file_path+'/'+file_name+'_mag.tif')
    
    img = Image.fromarray(data[2])
    img.save(file_path+'/'+file_name+'_phase.tif')
    
    img = Image.fromarray(data[3])
    img.save(file_path+'/'+file_name+'_real.tif')
    
    img = Image.fromarray(data[4])
    img.save(file_path+'/'+file_name+'_imag.tif')
    
    img = Image.fromarray(data[5])
    img.save(file_path+'/'+file_name+'_inclination.tif')    
    
def demo():

    lg.log_init()    

    file_path = r'E:\PhD\Finale Übergabe\EIM Zellen\Data Summary\245_EIM Zellen\mdck\MDCK 3\ag35 mdkc c7 pos2 CV05 647 short.tif'
    
    window_size = 10
    dt = 0.1
    frequency = 1.25
    t1 = datetime.now()
    
    data = image_fft(window_size, file_path, dt, frequency)
        
    t2 = datetime.now()
    delta_t = t2-t1
    
    print('time elapsed '+str(delta_t))
    print('now saving image...')
    save_image(data, file_path)
    print('done')

demo()
