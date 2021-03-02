
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

import os
import numpy as np
import pandas as pd
import inspect


def log_init():
    global logging
    global internal_logging
    global file
    global log_type
    global function_list

    function_list = []
    
    frame = inspect.stack()[1]
    path = inspect.getmodule(frame[0]).__file__
    
    file = path[:-3]+'_log.txt'
    text = 'start '+str(datetime.now())+'\n'
    f = open(file, "w")
    f.write(text)
    f.close()
    
    internal_logging = True
    logging = True
    return logging

def text_log(text):
        
    if 'logging' in globals():
        f = open(file, "a")
        
        max_len = 110
        
        if (len(text)>max_len and 'save plot' not in text):
            
            remaining = text
            
            while len(remaining)>max_len:
                
                i = max_len
 
                while remaining[i] not in [' ',',','.',':',';']:
                    i = i-1
                   
                text = remaining[0:i+1]
                f.write(text+'\n')
                
                if len(remaining[i+1:])<=max_len:
                    text = remaining[i+1:]
                    f.write(text+'\n')
                    break
                else:
                    remaining = remaining[i+1:]
        else:   
            f.write(text+'\n')
        
        f.close()

def var_log(var):
    if isinstance(var, tuple):
        var_list = var
        
        len_list = []
        df = pd.DataFrame()
        
        
        for item in var_list:
            len_list.append(len(item)) if hasattr(item, '__len__') else len_list.append(1)
        
        size = max(len_list)
        
        for i in range(0, len(var_list)):
            if hasattr(var_list[i], '__len__'):
                df.insert(i, str(i), list(var_list[i])+['']*(size-len(var_list[i])))
            else:
                df.insert(i, str(i), [var_list[i]]+['']*(size-1))
        
        f = open(file, "a")
        f.write(df.to_string(header = False, index = False))
        f.close()
        text_log('')
        return var_list
    else:
       text_log(str(var))
       return var

def function_log():
    
    parent_file = inspect.stack()[1][1].split(os.sep)[-1]
    parent_function = inspect.stack()[1][3]
    
    grandparent_file = inspect.stack()[2][1].split(os.sep)[-1]
    grandparent_function = inspect.stack()[2][3]
    
    text = grandparent_function + '('+grandparent_file+')' +' --> '+ parent_function+ '('+parent_file+')...'
    text_log(text)
    
    """
    if logging == 'report':
        if text not in function_list:
            function_list.append(text)
            internal_logging = True
        else:
            internal_logging = False
            function_queue = getattr()
    """        

def data_type(data):
    
    types = [list, tuple, np.ndarray]
    
    result = [x for x in types if type(data) == x]
    
    if len(result) == 0:
        return None
    else:
        if result[0].__name__ == 'ndarray':
            return 'list'
        else:
            return result[0].__name__

def plot_type(data):
    c_type = [data_type(data), data_type(data[0]), data_type(data[1])]
    
    y = [['list', None, None], 'y']
    x_y = [['tuple', 'list', 'list'], 'x_y']
    x_yn = [['tuple', 'list', 'tuple'], 'x_yn']
    xn_yn = [['tuple', 'tuple', 'tuple'], 'xn_yn']
    
    type_list = [y, x_y, x_yn, xn_yn]
    for item in type_list:
        if (c_type[0] == item[0][0] and c_type[1] == item[0][1] and c_type[2] == item[0][2]):
            return item[1]

def img_log(data, title=None, plot=None, legend=None, x_axis=None, y_axis=None):
    
    #automatic recognition if scatter or line by analysing x values (strict increase -> line, else scatter)
    #fig = plt.figure(figsize=(10, 5))
    
    if 'logging' in globals():
        plt.title(title)
        
        if plot == None:
            p_type = plot_type(data)
            
            if p_type == 'y':
                plt.plot(data, label = legend)
                
            if p_type == 'x_y':
                plt.plot(data[0], data[1], label = legend)
                
            if p_type == 'x_yn':
                for i in range(0,len(data[1])):
                    plt.plot(data[0], data[1][i], label = legend[i])
                    
            if p_type == 'xn_yn':
                for i in range(0,len(data)):
                    plt.plot(data[i][0], data[i][1], label = legend[i])
        else:
            if plot == 'yn':
                for i in range(0,len(data)):
                    plt.plot(data[i], label = legend[i])
                    
            if plot == 'heatmap':
                sns.heatmap(data, xticklabels=False, yticklabels=False)
        
        
        path = file[:-4]+'_'
        file_name = path+title+'_'+str(datetime.now()).replace(':','').replace('.','').replace('-','') +'.jpg'
        text_log('save plot -> '+file_name)
        
        plt.legend()
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.grid()
        plt.savefig(file_name)
        
        #fig.savefig(title, dpi=600)
        
        plt.close('all')
    else:
        plt.close('all')

def pdf_report():
    if 'logging' in globals():
        
        f = open(file, "r")
        f.close()
        
def demo(): 


    log_init()
    
    
    year = [2014, 2015, 2016, 2017, 2018, 2019]
    count = [39, 117, 111, 110, 67, 29]
    year2 = [2004, 2011, 2015, 2018, 2025, 2030]
    count2 = [234, 517, 1231, 10, 63, 34]
    
    
    #plt.plot(year, tutorial_count, color="#6c3376", linewidth=3) 
    #plt.xlabel('Year')  
    #plt.ylabel('Number of futurestud.io Tutorials')
    img_log(((year, count), (year2, count2)), title = 'test', legend=['c1', 'c2'], x_axis = 'year', y_axis = 'count')

sns.heatmap
    

