import numpy as np
import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt 
import csv
import itertools
import statsmodels
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from IPython.display import clear_output
from sklearn.neighbors import KernelDensity

def largest_prime_factor(n: int):
  '''plotting tool for subplots'''

  prime_fact = 1
  i = 2 #Initial_cond

  while i <= n / i:
    if n % i == 0:
      prime_fact = i
      n = n / i
     
    else:
        i += 1 
  if prime_fact < n:
    prime_fact = int(n)
  
  return prime_fact 

def import_fbg_data(filepath: str):
  with open(filepath) as f:
    count = 0 
    for line in f:
      if line[0:4] == 'Time' and not line[0:5] == 'Times':
        break
      else:
        count = count + 1

  data = pd.read_csv(filepath, skiprows = count+1, sep='\t', header = None)
  data = data.drop(data.columns[[2, 3, 4]], axis=1)
  data = data.rename(columns={0: "Time", 1: "Number of Sensors"})
  max_vals = data['Number of Sensors'].max()
 
  hmm = {}
  for i in range(0,max_vals):
    hmm[5 + i] = f"{i}"
   
  data = data.rename(columns=hmm)
  data = data.to_dict(orient='list')

  return data

def import_data(filepath: str, convert_to_g = False):
  '''Loads any .txt data tab seperated, where the first line
  is the keys, and the latter lines are the data recovered. 

  Returns:
    data - A dictionary containing the data and the keys from 
    the file. 
  '''

  with open(filepath) as f:
    reader = csv.reader(f, delimiter='\t')
    d = list(reader)
  
  data = {}
  keys = d[0]

  for i in range(len(keys)):
    if not convert_to_g:
      data[keys[i]] = np.array([float(x) for x in np.array(d).T[i][1:]])
    else:
      sensitivity = convert_to_g[0] #mV/g units
      ranges = convert_to_g[1]
      data[keys[i]] = np.array(([float(x)*1000/(sensitivity) - ranges for x in np.array(d).T[i][1:]]))
      #Multiply by 1000 to convert V to mV, then factor out the sensitivy to give g calculation
  return data


def detrend_data(data: dict):
  '''
  Detrends the data, by a linear least-squares fit to the data, subracted
  from the data.

  Returns:
    detrended_dict - Detrended data, with the same keys as the original data
                     with detrended values.

  '''
  detrended_dict = {}
  for i in data.keys():
    detrended_dict[i] = sp.signal.detrend(data[i])

  return detrended_dict

def filter_data(data: dict, order, cutoff, bandpass, sample_frequency,plot_response=False):
  b, a = sp.signal.butter(order, cutoff, btype = bandpass, fs = sample_frequency)
  if plot_response:
    fig, ax = plt.subplots(3,1,figsize=(16,9))
    [w, h] = sp.signal.freqz(b, a, fs=sample_frequency)
    ax[0].plot(w,np.abs(h))
    ax[1].plot(w,np.real(h))
    ax[2].plot(w,np.imag(h))
    ax[0].set_title("Magnitude")
    ax[1].set_title("Real")
    ax[2].set_title("Imaginary")
  filtered_dict = {}
  for i in data.keys():
    input = data[i]
    output = sp.signal.filtfilt(b, a, input)
    filtered_dict[i] = output
  
  return filtered_dict

def power_spectrum(data, sample_rate,to_disp=False):
  power_dict = {}
  for key in data.keys():
    input = data[key]
    length = len(data[key])

    fft = np.fft.fft(input) / length 
    nyquist_frequency = sample_rate / 2

    freqs = nyquist_frequency * np.linspace(0,1,int(length/2)+1)
    base_res = fft[0:int(length/2)+1]
    if to_disp:
      base_res = base_res / -(2*np.pi*freqs)**2 #Definition.
    power_dict[key] = [freqs,2*np.abs(base_res)**2]
    

  return power_dict

def set_sample_rate(sample_rate: int):
  return sample_rate


def cross_correlate(data1, data2):
  cross_corr = np.real(np.fft.fftshift(np.fft.ifft(np.fft.fft(data1) * 
              np.fft.fft(data2[::-1]))))
  return cross_corr

def calculate_coherence(data1, data2, sample_rate):
  f, coh = sp.signal.coherence(data1, data2, fs=sample_rate)
  return [f, coh]


def calculate_lags(data1, data2):
  lags = np.arange(-len(data2)+1, len(data1))
  mid = lags.size // 2    
  lag_bound = len(data1) // 2

  if len(data1) % 2 == 0:
    lags = lags[(mid-lag_bound):(mid+lag_bound)]
  else:
    lags = lags[(mid-lag_bound):(mid+lag_bound)+1]

  return lags

def plot_time_series(data: dict, keys: list = False, title = False, lim = False):
  '''Plots the time series for the data, either a subset or the full
  array.'''
  if not keys:
    number_of_plots = len(data.keys())
    prime_fact = largest_prime_factor(number_of_plots)
    [width , height] = [prime_fact , 
                        int(number_of_plots / prime_fact)]

    temp_keys = []
    for key in data.keys():
      temp_keys.append(key)

    fig, ax = plt.subplots(width, height) 
    fig.set_figheight(6 * height)
    fig.set_figwidth(8 * width)

    for i in range(number_of_plots):
      [x, y] = [i % prime_fact, i // prime_fact] 
      ax[x,y].plot(data[temp_keys[i]])
      ax[x,y].set_title(temp_keys[i])
      if lim:
        ax[x,y].set_xlim(lim[0],lim[1])
    
  else:
    number_of_plots = len(keys)
    prime_fact = largest_prime_factor(number_of_plots)
    [width, height] = [prime_fact, 
                       int(number_of_plots / prime_fact)]

    fig, ax = plt.subplots(width, height)
    fig.set_figheight(6 * height)
    fig.set_figwidth(8 * width)

    for i in range(number_of_plots):
      [x, y] = [i % prime_fact, i // prime_fact]
      ax[x,y].plot(data[keys[i]])
      ax[x,y].set_title(keys[i])
      if lim:
        ax[x,y].set_xlim(lim[0],lim[1])

  if not title:
    fig.suptitle("Time series plots for the data")
  else:
    fig.suptitle(title)

def plot_power_spectrum(data, keys = False, logs = False, lims = False, title = False):
  if not keys:
    number_of_plots = len(data.keys())
    prime_fact = largest_prime_factor(number_of_plots)
    [width, height] = [prime_fact, int(number_of_plots / prime_fact)]

    temp_keys = []
    for key in data.keys():
      temp_keys.append(key)

    fig, ax = plt.subplots(width, height)
    fig.set_figheight(6 * height)
    fig.set_figwidth(8 * width)

    for i in range(number_of_plots):
      [x, y] = [i % prime_fact, i // prime_fact]
      if logs:
        ax[x, y].plot(data[temp_keys[i]][0], np.log(data[temp_keys[i]][1]))
        ax[x,y].set_title(temp_keys[i])
      else:
        ax[x, y].plot(data[temp_keys[i]][0], data[temp_keys[i]][1])
        ax[x,y].set_title(temp_keys[i])
      
      if lims:
        ax[x, y].set_xlim(lims[0], lims[1])

    if not title:
      fig.suptitle("Power spectrum for the data")
    else:
      fig.suptitle(title)

  else:
    number_of_plots = len(keys)
    temp_keys = keys
    prime_fact = largest_prime_factor(number_of_plots)
    [width, height] = [prime_fact, 
                       int(number_of_plots / prime_fact)]

    fig, ax = plt.subplots(width, height)
    fig.set_figheight(6 * height)
    fig.set_figwidth(8 * width)

    for i in range(number_of_plots):
      [x, y] = [i % prime_fact, i // prime_fact]
      if logs:
        ax[x, y].plot(data[temp_keys[i]][0], np.log(data[temp_keys[i]][1]))
        ax[x,y].set_title(temp_keys[i])
      else:
        ax[x, y].plot(data[temp_keys[i]][0], data[temp_keys[i]][1])
        ax[x,y].set_title(temp_keys[i])
      
      if lims:
        ax[x, y].set_xlim(lims[0], lims[1])

    if not title:
      fig.suptitle("Power spectrum for the data")
    else:
      fig.suptitle(title)

def plot_power_spectrum_density(data, SAMPLE_RATE, keys = False, lims = False, 
                                title = False, windows='hann', npersegs=None, noverlaps=None, nffts=None,
                                detrends='constant', return_onesideds=True, scalings='density', axiss= -1,
                                averages='mean'):
  if not keys:
    number_of_plots = len(data.keys())
    prime_fact = largest_prime_factor(number_of_plots)
    [width, height] = [prime_fact, int(number_of_plots / prime_fact)]

    temp_keys = []
    for key in data.keys():
      temp_keys.append(key)

    fig, ax = plt.subplots(width, height)
    fig.set_figheight(90 * height)
    fig.set_figwidth(8 * width)

    for i in range(number_of_plots):
      [x, y] = [i % prime_fact, i // prime_fact]
      f, res = sp.signal.welch(data[temp_keys[i]],SAMPLE_RATE,window = windows,
                               nperseg = npersegs, noverlap = noverlaps,nfft=nffts,
                               detrend=detrends, return_onesided = return_onesideds, scaling = scalings,
                               axis = axiss, average = averages)
      ax[x, y].plot(f, 10*np.log10(res))
      ax[x,y].set_title(temp_keys[i])
      ax[x,y].set_ylabel("PSD (dB/Hz)")
      ax[x,y].grid()
      
      if lims:
        ax[x, y].set_xlim(lims[0], lims[1])

    if not title:
      fig.suptitle("Power spectrum density for the data")
    else:
      fig.suptitle(title)

  else:
    number_of_plots = len(keys)
    temp_keys = keys
    prime_fact = largest_prime_factor(number_of_plots)
    [width, height] = [prime_fact, 
                       int(number_of_plots / prime_fact)]

    fig, ax = plt.subplots(width, height)
    fig.set_figheight(18 * height)
    fig.set_figwidth(8 * width)

    for i in range(number_of_plots):
      [x, y] = [i % prime_fact, i // prime_fact]
      f, res = sp.signal.welch(data[temp_keys[i]],SAMPLE_RATE,window = windows,
                               nperseg = npersegs, noverlap = noverlaps,nfft=nffts,
                               detrend=detrends, return_onesided = return_onesideds, scaling = scalings,
                               axis = axiss, average = averages)
      ax[x, y].plot(f, 10*np.log10(res))
      ax[x,y].set_title(temp_keys[i])
      ax[x,y].set_ylabel("PSD (dB/Hz)")
      ax[x,y].grid()
      if lims:
        ax[x, y].set_xlim(lims[0], lims[1])

    if not title:
      fig.suptitle("Power spectrum for the data")
    else:
      fig.suptitle(title)

def plot_kde(data: dict, keys = False,title=False):
  if not keys:
    number_of_plots = len(data.keys())
    prime_fact = largest_prime_factor(number_of_plots)
    [width, height] = [prime_fact, int(number_of_plots / prime_fact)]

    temp_keys = []
    for key in data.keys():
      temp_keys.append(key)

    fig, ax = plt.subplots(width, height)
    fig.set_figheight(6 * height)
    fig.set_figwidth(8 * width)

    for i in range(number_of_plots):
      [x, y] = [i % prime_fact, i // prime_fact]
      x_test = np.linspace(np.min(data[temp_keys[i]]),np.max(data[temp_keys[i]]),1000)
      kde_ml = scipy.stats.gaussian_kde(data[temp_keys[i]])
      ax[x, y].plot(x_test, kde_ml(x_test))
      ax[x, y].hist(data[temp_keys[i]],bins=100,density=True,stacked=True)
      ax[x,y].set_title(temp_keys[i])
      ax[x,y].grid()

    if not title:
      fig.suptitle("Kernel Density estimation + histogram for the data")
    else:
      fig.suptitle(title)

  else:
    number_of_plots = len(keys)
    temp_keys = keys
    prime_fact = largest_prime_factor(number_of_plots)
    [width, height] = [prime_fact, 
                       int(number_of_plots / prime_fact)]

    fig, ax = plt.subplots(width, height)
    fig.set_figheight(6 * height)
    fig.set_figwidth(8 * width)

    for i in range(number_of_plots):
      [x, y] = [i % prime_fact, i // prime_fact]
      x_test = np.linspace(np.min(data[temp_keys[i]]),np.max(data[temp_keys[i]]),1000)
      kde_ml = scipy.stats.gaussian_kde(data[temp_keys[i]])
      ax[x, y].plot(x_test, kde_ml(x_test))
      ax[x, y].hist(data[temp_keys[i]],bins=100,density=True,stacked=True)
      ax[x,y].set_title(temp_keys[i])
      ax[x,y].grid()

    if not title:
      fig.suptitle("Kernel Density estimation + histogram for the data")
    else:
      fig.suptitle(title)

def cross_correlation(data: dict, keys: list = False, norm:bool = False,
                    plot:bool = False,include_sample_rate = False):
  '''
  Cross correlation of all keys against all other keys, in a triangle plot if None is selected.
  Else, the keys listed are all cross correlated to one another. 

  The cross correlation has been computed from the definiton of convolution,
  where the fourier transform of convolution is a multiplication, and time reversal
  is defined as complex conjugate. 
  '''
  if len(keys) == 2:
    if norm:
      cross_corr = cross_correlate(data[keys[0]]/np.std(data[keys[0]]), 
                                   data[keys[1]]/np.std(data[keys[1]])) 
              
      cross_corr = cross_corr / len(data[keys[0]]) #Normalising factor.
      lags = calculate_lags(data[keys[0]], data[keys[1]])
      
      if include_sample_rate:
        lags = lags / include_sample_rate
      
      if plot:
        plt.plot(lags,cross_corr)
        plt.show()
      return cross_corr
  

    else:
      cross_corr = cross_correlate(data[keys[0]], 
                                   data[keys[1]]) 
      
      lags = calculate_lags(data[keys[0]], data[keys[1]])
      
      if include_sample_rate:
        lags = lags / include_sample_rate

      if plot:
        plt.plot(lags, cross_corr)
        plt.show()
      return cross_corr
  
  else:
    cross_corr = []
    if not keys:
      temp_keys = []
      for key in data.keys():
        temp_keys.append(key)
    
    else:
      temp_keys = keys
    
    number_of_datas = len(temp_keys) #Number of cross correlations.
    combinations = list(itertools.combinations_with_replacement(temp_keys, 2))
    
    for combo in combinations:
      if not norm:
        cross_corr.append(cross_correlate(data[combo[0]],
                        data[combo[1]]))

      else:
        cross_correl = cross_correlate(data[combo[0]]/np.std(data[combo[0]]), 
                                   data[combo[1]]/np.std(data[combo[1]])) 
              
        cross_correl = cross_correl / len(data[combo[0]]) #Normalising factor.

        cross_corr.append(cross_correl)
    
    lags = calculate_lags(data[temp_keys[0]], data[temp_keys[1]])
      
    if include_sample_rate:
      lags = lags / include_sample_rate

    tri = np.zeros((len(temp_keys), len(temp_keys)))
    tri[np.triu_indices(len(temp_keys))] = 1

    [width , height] = [len(temp_keys) , 
                        len(temp_keys)]

    fig, ax = plt.subplots(width, height) 
    fig.set_figheight(6 * height)
    fig.set_figwidth(8 * width)

    count = 0 
    for i in range(number_of_datas**2):
      [x, y] = [i % number_of_datas, i // number_of_datas] 
      if tri[y, x] == 1:
        ax[x,y].plot(lags, cross_corr[count])
        count = count + 1
      else:
        ax[x,y].axis('off')
    #  ax[x,y].set_title(temp_keys[i])

    return cross_corr

def coherence(data: dict, keys: list = False,
                    plot:bool = False,include_sample_rate = None):
  
  if len(keys) == 2:
    f, coh = calculate_coherence(data[keys[0]], 
                                 data[keys[1]],include_sample_rate)       
    if plot:
      plt.plot(f,coh)
      plt.show()
    return coh
  
  else:
    coh_list = []
    if not keys:
      temp_keys = []
      for key in data.keys():
        temp_keys.append(key)
    
    else:
      temp_keys = keys
    
    number_of_datas = len(temp_keys) #Number of cross correlations.
    combinations = list(itertools.combinations_with_replacement(temp_keys, 2))
    
    for combo in combinations:
      f, coh = calculate_coherence(data[combo[0]], 
                                 data[combo[1]],include_sample_rate) 
              
      coh_list.append(coh)
    
    tri = np.zeros((len(temp_keys), len(temp_keys)))
    tri[np.triu_indices(len(temp_keys))] = 1

    [width , height] = [len(temp_keys) , 
                        len(temp_keys)]

    fig, ax = plt.subplots(width, height) 
    fig.set_figheight(6 * height)
    fig.set_figwidth(8 * width)

    count = 0 
    for i in range(number_of_datas**2):
      [x, y] = [i % number_of_datas, i // number_of_datas] 
      if tri[y, x] == 1:
        ax[x,y].plot(f, coh_list[count])
        count = count + 1
      else:
        ax[x,y].axis('off')
    #  ax[x,y].set_title(temp_keys[i])

    return coh_list

def arima(data,order = (2,2,2)):
  ''' 
  @MISC {pmdarima,
  author = {Taylor G. Smith and others},
  title  = {{pmdarima}: ARIMA estimators for {Python}},
  year   = {2017--},
  url    = "http://www.alkaline-ml.com/pmdarima",
  note   = {[Online; accessed <today>]}
 }
  '''
  from pmdarima.arima import auto_arima
  model = auto_arima(data, start_p=1, start_q=1,
                           max_p=12, max_q=12, m=1,
                           seasonal=False,
                           d=0, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
  fig, ax = plt.subplots(1)
  fit = model.predict_in_sample()
  print(model.summary())
  ax.plot(fit)
  ax.plot(data)
  return model

def pca(data):
  '''
  Machine learning PCA :tm:
  '''
  from sklearn.decomposition import PCA
  pca2 = PCA(n_components=2)
  reduced = pca2.fit(np.array(data).reshape(1,-1))
  print(reduced.shape)
  return reduced

def calculate_statistics(data):
  '''Calculate various statistics for the time series data. The following being
  Max,Min,Mean,Standard Deviation, variance, RMS, Autocorrelation with different lags?'''
  statistics_dict = {}
  for key in data.keys():
    statistics_dict[key] = {'mean': np.mean(data[key]),
                          'max': np.max(data[key]),
                          'min': np.min(data[key]),
                          'std' : np.std(data[key]),
                          'RMS' : np.sqrt(np.mean(data[key]**2))}
  return statistics_dict

def white_noise_checker(data):
  '''Important to check because it's not predictable by definition'''
  ljung_box = {}
  for key in data.keys():
    ljung_box[key] = acorr_ljungbox(data[key])
  
  return ljung_box

def turbulent_signal_checker(power,filters=[0,20]):
  """
  Uses the first order linear regression within an interrogation range to estimate 
  the roll-off in loglog scale. Apperently we should see something like -5/7
  """
  slopes = {}
  for key in power.keys():
    f = power[key][0]
    idx = np.where((f >= filters[0]) & (f <= filters[1]))
    ex = power[key][0][idx][1:]
    ps =  power[key][1][idx][1:]
    X = np.zeros((2,len(ex))).T + 1
    X.T[1] = np.log(ex)
    y = np.log(ps).T

    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    slopes[key] = beta[1]
  return slopes

  
