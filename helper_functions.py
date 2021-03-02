



def generate_bounds(d_min, d_max):

    bounds_min = []
    bounds_max = []
    
    for i in d_min:
        bounds_min.append(d_min)
        bounds_max.append(d_max)
               
    bnds = ((*bounds_min,), (*bounds_max,))
    
    return bnds
            
def bounds_merge(merged_bounds):
    
    bounds_min = []
    bounds_max = []
    
    for i in merged_bounds:
        bounds_min = bounds_min + (list(i[0]))
        bounds_max = bounds_max + (list(i[1]))
               
    bnds = ((*bounds_min,), (*bounds_max,))
    
    return bnds

def fit_params(params):
    p0 = []
    bounds_min = []
    bounds_max = []
    
    for p in params:
        p0.append(p[0])
        bounds_min.append(p[1])
        bounds_max.append(p[2])
    
    bnds = ((*bounds_min,), (*bounds_max,))
    
    return p0, bnds
    
