import math
import warnings
import numpy as np
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')


def intersection(c, r, e1, e2, z):
    c = math.radians(c)
    e2 = math.radians(e2)
    z = math.radians(z)

    a = e1*math.cos(e2) - math.cos(c)
    b = e1*math.sin(e2) - math.sin(c)

    roots = np.roots([1, 2*(a*math.cos(z) + b*math.sin(z)), a**2 + b**2 - r**2])
    root = max(roots)
    point = (e1*math.cos(e2) + root*math.cos(z), e1*math.sin(e2) + root*math.sin(z))
    
    return point


def bestR(c, e1, e2, z, s, times, oppositions):
    equant_longitudes = [(z + s * i) % 360 for i in times]
    radius = 0

    for i in range(len(times)):
        m1 = math.tan(math.radians(oppositions[i]))
        m2 = math.tan(math.radians(equant_longitudes[i]))
        
        x = (e1*math.sin(math.radians(e2)) - m2*e1*math.cos(math.radians(e2))) / (m1 - m2)
        y = m1*x

        radius += math.sqrt((x - math.cos(math.radians(c)))**2 + (y - math.sin(math.radians(c)))**2)
    
    return radius / len(times)


def MarsEquantModel(c, r, e1, e2, z, s, times, oppositions):
    errors = []
    equant_longitudes = [(z + s * i) % 360 for i in times]
        
    for i in range(len(times)):
        equant_point = intersection(c, r, e1, e2, equant_longitudes[i])
        equant_angle = math.degrees(math.atan2(equant_point[1], equant_point[0])) % 360
        
        delta = equant_angle - oppositions[i]
        if delta > 180:
            delta -= 360
        elif delta < -180:
            delta += 360
        errors.append(delta)
    
    emax = max(np.abs(np.array(errors)))

    return errors, emax


def filter(e1, s, times, oppositions, threshold=0.5):
    params = []

    for c in np.arange(0, 360, 10):
        for e2 in np.arange(0, 360, 10):
            z_start = int(oppositions[0])
            z_end = (e2 - 180) % 360

            if z_end - z_start > 180:
                z_end -= 360
            
            if z_start > z_end:
                z_start, z_end = z_end, z_start

            for z in np.arange(z_start, z_end, 1):
                r_average = bestR(c, e1, e2, z, s, times, oppositions)

                for r in np.arange(r_average*0.9, r_average*1.1, r_average*0.01):
                    max_error = MarsEquantModel(c, r, e1, e2, z, s, times, oppositions)[1]
                    if max_error < threshold:
                        params.append((c, r, e1, e2, z, s))
    
    return params


def finetune(c, r, e1, e2, z, s, times, oppositions, threshold=1/3600):
    delta_start = math.inf
    errors, delta_end = MarsEquantModel(c, r, e1, e2, z, s, times, oppositions)
    min_value = sum(np.square(errors))
    n = 0
    
    while delta_start - delta_end > threshold:
        delta_start = delta_end
        
        for c_new in np.arange(c - 5/2**n, c + 5/2**n, 1/2**n * 0.01):
            errors = MarsEquantModel(c_new, r, e1, e2, z, s, times, oppositions)[0]
            value = sum(np.square(errors))
            
            if value < min_value:
                min_value = value
                c = c_new

        for r_new in np.arange(r - (r*0.05)/2**n, r + (r*0.05)/2**n, r/2**n * 0.0001):
            errors = MarsEquantModel(c, r_new, e1, e2, z, s, times, oppositions)[0]
            value = sum(np.square(errors))
            
            if value < min_value:
                min_value = value
                r = r_new
        
        for e1_new in np.arange(e1 - (e1*0.05)/2**n, e1 + (e1*0.05)/2**n, e1/2**n * 0.0001):
            errors = MarsEquantModel(c, r, e1_new, e2, z, s, times, oppositions)[0]
            value = sum(np.square(errors))
            
            if value < min_value:
                min_value = value
                e1 = e1_new
        
        for e2_new in np.arange(e2 - 5/2**n, e2 + 5/2**n, 1/2**n * 0.01):
            errors = MarsEquantModel(c, r, e1, e2_new, z, s, times, oppositions)[0]
            value = sum(np.square(errors))

            if value < min_value:
                min_value = value
                e2 = e2_new

        for z_new in np.arange(z - 0.5/2**n, z + 0.5/2**n, 1/2**n * 0.001):
            errors = MarsEquantModel(c, r, e1, e2, z_new, s, times, oppositions)[0]
            value = sum(np.square(errors))
            
            if value < min_value:
                min_value = value
                z = z_new
        
        for s_new in np.arange(s - (s*0.005)/2**n, s + (s*0.005)/2**n, s/2**n * 0.00001):
            errors = MarsEquantModel(c, r, e1, e2, z, s_new, times, oppositions)[0]
            value = sum(np.square(errors))
            
            if value < min_value:
                min_value = value
                s = s_new
        
        delta_end = MarsEquantModel(c, r, e1, e2, z, s, times, oppositions)[1]
        n += 1
    
    return (c, r, e1, e2, z, s, delta_end)


def bestMarsOrbitParams(times, oppositions, threshold=1/15, maxiter=5):
    s = 360/687
    cpus = cpu_count() - 1
    best_fit = math.inf
    best_params = ()
    iter = 0

    while best_fit > threshold and iter < maxiter:
        with Pool(cpus) as pool:
            e1_range = np.arange(0.5 + 0.1*cpus*iter, 0.5 + 0.1*cpus*(1 + iter), 0.1)
            params = pool.starmap(filter, [(e1, s, times, oppositions) for e1 in e1_range])
        
        params = [sublist for sublist in params if sublist]
        if len(params) == 0:
            iter += 1
            continue
        
        params = np.concatenate(params)
        np.random.shuffle(params)
        
        with Pool(cpus) as pool:
            final = pool.starmap(finetune, [(*param, times, oppositions) for param in params])
        
        final = np.array(final)
        final = final[final[:,6].argsort()]
        new_fit = final[0][6]
        
        if new_fit < best_fit:
            best_fit = new_fit
            best_params = final[0][:6]
        
        iter += 1
        
    errors, maxError = MarsEquantModel(*best_params, times, oppositions)
    r, s, c, e1, e2, z = best_params[1], best_params[5], best_params[0], best_params[2], best_params[3], best_params[4]
    
    return r, s, c, e1, e2, z, errors, maxError
