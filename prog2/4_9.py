#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/py
import sklearn
import logging
import os
import warnings
import numpy as np
import tsfel
from collections import namedtuple
from contextlib import contextmanager
from datetime import datetime
from zipfile import ZipFile, is_zipfile
import matplotlib.pyplot as plt
import pandas as pd
import math
import tsfresh.feature_extraction.feature_calculators as feature
from tsfresh.utilities.distribution import MultiprocessingDistributor
import seaborn as sns
from datetime import date


# In[2]:


HEADER_LINE_PREFIX = '% '
HEADER_END_LINE_PREFIX = '%-'
HEADER_KEYS = ['date_time', 'prog_version', 'serial', 'mech_unit', 'axis', 'sampling_period', 'tool', 'rob_ax', 'ext_ax']
DATETIME_FORMAT = '%Y%m%d_%H%M%S'

MIN_SAMPLES = 600
SAMPLING_PERIOD = 0.004032
FREQUENCY = 1/SAMPLING_PERIOD

# Retrieves a pre-defined feature configuration file to extract all available features
cfg = tsfel.get_features_by_domain()


# In[3]:


MccRun = namedtuple('MccRun', 'mode series')
MccResult = namedtuple('MccResult', 'status value average distance')
OperationMode = namedtuple('OperationMode', ['axis', 'mech_unit', 'tool', 'rob_ax', 'ext_ax', 'prog_version'])


# In[4]:


@contextmanager
def open_zip(file):
    if not is_zipfile(file):
        raise TypeError(f'{file} is not a valid zip file')

    zip_archive = ZipFile(file, 'r')
    files = zip_archive.namelist()
    try:
        extracted = [zip_archive.open(f) for f in files]
        yield extracted
    finally:
        for f in extracted:
            f.close()
        zip_archive.close()


# In[5]:


def read_header(file):
    header = {}
    for line in file:
        line = line.decode('utf-8')
        if line.startswith(HEADER_LINE_PREFIX):
    
            parameter, value = (element.strip() for element in line.split(':'))
            if 'Meas time' in parameter:
                header['date_time'] = datetime.strptime(value, DATETIME_FORMAT)
            if 'Program version' in parameter:
                header['prog_version'] = value
            elif 'Robot serial number' in parameter:
                header['serial'] = value
            elif 'Mech unit' in parameter:
                header['mech_unit'] = value
            elif 'Axis' in parameter:
                header['axis'] = int(value)
            elif 'Sample time' in parameter:
                header['sampling_period'] = float(value)
            elif 'Tool' in parameter:
                header['tool'] = value
            elif 'RobAx' in parameter:
                header['rob_ax'] = value
            elif 'ExtAx' in parameter:
                header['ext_ax'] = value
            elif 'Meas type' in parameter:
                continue
            else:
                #logging.debug(f'Parameter {parameter} and value {value} not handled')
                continue

        elif line.startswith(HEADER_END_LINE_PREFIX):
            continue

        else:
            break
#     if not header:
#         logging.warning(f'Empty header in {file.name}')
#         return None
    return header


# In[6]:


def read_data(file):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            data = np.loadtxt(file, dtype=np.float64, delimiter='\t')
        except Warning:
            logging.warning(f'{file.name} has no data')
            return None
#         except Exception:
#             logging.error(f'{file.name} is broken')
            return None

    n_samples, n_columns = data.shape

    if n_columns == 3:
        data = _decrypt3(data)
    elif n_columns == 4:
        data = _decrypt3(data[:, 1:]) 
    elif n_columns == 5:
        data = _decrypt4(data[:, 1:])
    else:
        logging.warning(f'Unhandled case with {n_columns} columns in {handle.name}')
        return None
    if n_samples < MIN_SAMPLES:
        logging.warning(f'Case with {n_samples} samples')

    data = data[10:-10]
    return np.asarray(data)
def _decrypt3(data):
    return data * np.matrix([[0.5, 0.5, 0], [-0.5, 0, 0.5], [0, 0.5, -0.5]])


def _decrypt4(data):
    return data * np.matrix([[0.5, 0.5, 0, 0], [-0.5, 0, 0, 0.5], [0, 0, 0.5, -0.5], [0, 0.5, -0.5, 0]])


# In[7]:



class MccTimeSeries():
    def __init__(self, series, date_time, sampling_period):
        self.series = series
        self.date_time = date_time
        self.sampling_period = sampling_period
    
    def __getitem__(self, index):
        return self.series[index]

    def __len__(self):
        return len(self.series)

    @property
    def torque_ff(self):
        return self.series[:, 0]

    @property
    def velocity(self):
        return self.series[:, 1]
    
    @property
    def torque(self):
        return self.series[:, 2]

    @property
    def time_index(self):
        return np.arange(0, self.torque.size)[:, np.newaxis] * self.sampling_period

    @property
    def sampling_frequency(self):
        return 1 / self.sampling_period

    def normalized_torque(self):
        return (self.torque - self.torque.min()) / (self.torque.max() - self.torque.min())
    
    def standardized_torque(self):
        return (self.torque - self.torque.mean()) / self.torque.std()
    def mean_torque(self):
        return self.torque.mean()
    
    def find_reference_point(self):
        ref = self.standardized_torque()
        mean_value = self.torque.mean()
        stv = self.torque.std(ddof=0)
        return ref, mean_value, stv
        
    
    


# # Get the reference points

# # Failure Case 2

# In[8]:


axes = ['axis1', 'axis2', 'axis3', 'axis4', 'axis5', 'axis6']

def process_file(file_path):
    ref = {}
    mean = {}
    stv = {}

    with ZipFile(file_path) as File:
        print(File)
        for file in File.namelist():
            with File.open(file) as f:
                header = read_header(f)
                data = read_data(f)
                mode = OperationMode(axis=header['axis'], mech_unit=header['mech_unit'], tool=header['tool'], rob_ax=header['rob_ax'], ext_ax=header['ext_ax'], prog_version=header['prog_version'])
                series = MccTimeSeries(data, header['date_time'], header.get('sampling_period', SAMPLING_PERIOD))                
                for ax in axes:
                    if ax in file:
                        ref[ax], mean[ax], stv[ax] = series.find_reference_point()
                        

        return ref, mean, stv


# In[9]:


# Use the standard deviation and the mean of reference points
def standardized_torque(data,ref_mean,ref_stv):
    return (data - ref_mean) / ref_stv


# In[10]:


# Caculate the corss-correlation of two consecutive measurements
from scipy import signal
def cross_correlation(ref,d):
    Shift = []
    corr = signal.correlate(ref,d)
#Find the index of the maximum of cross-correlation
    indx = np.argmax(corr, axis=0)
#Offset of the two series
    lags = signal.correlation_lags(len(ref), len(d))
    Shift.append(lags[indx])
    offset = lags[indx]
    min_length = min(d.size,ref.size)-abs(offset)
    d_m, ref_m = shift_signal(offset, min_length, d, ref)
    return d_m, ref_m, offset


# In[11]:


def shift_signal(offset, min_length, d, ref):    
    if offset > 0:
        d = d[:min_length] ;
        ref = ref[offset:];
    elif offset < 0:
        d = d[-offset:];
        ref = ref[:min_length]
    return d,ref


# In[12]:


def trimmedMean(mcc_torque):   
    mcc_torque = mcc_torque[np.abs(mcc_torque) > 1e-4]
    try:
          return mcc_torque.mean()      
    except:
          return null


# In[13]:


def torque_fftFeatures_maxamplitude(mcc_torque):   
    amplitudes = np.fft.fft(mcc_torque)
    realAmplitudes = abs(amplitudes)  # real valued amplitudes
    maxAmplitude = np.max(realAmplitudes)
    return maxAmplitude


# In[14]:


def torque_fftFeatures_entropy(mcc_torque):
      try:  
            amplitudes = np.fft.fft(mcc_torque)
            realAmplitudes = abs(amplitudes)  # real valued amplitudes
            N = mcc_torque.shape[0]
            entropy = 0
            for i in realAmplitudes:
                  try:
                        entropy += i * log(i)
                  except:
                        entropy += 0
            entropy = -entropy / np.float(log(N))
            return entropy
      except:
            return 0           


# In[15]:


def area_signal(d, ref, ref_mean, dx):
    abs(np.trapz(abs(d), axis = ref_mean, dx=dx) - np.trapz(abs(ref), axis = ref_mean, dx=dx))


# In[16]:


def generate_more_feature(Filepath,ref,mean,stv):    
    File_path = Filepath
    ax1,ax2,ax3,ax4,ax5,ax6 = 'axis1','axis2','axis3','axis4','axis5','axis6'
    columns=['axis1','axis2','axis3','axis4','axis5','axis6']
    RMS_Value=pd.DataFrame(columns=columns)
    Skewness_value = pd.DataFrame(columns=columns)
    Kurtosis_value = pd.DataFrame(columns=columns)
    
    Energy = pd.DataFrame(columns=columns)
    Stv = pd.DataFrame(columns=columns)
    Variance = pd.DataFrame(columns=columns)
    Impulse_factor = pd.DataFrame(columns=columns)
    Shape_factor = pd.DataFrame(columns=columns)
    MAX_AMP = pd.DataFrame(columns=columns)
    torque_areaUnderSignal = pd.DataFrame(columns=columns)
    torque_trimmedMean = pd.DataFrame(columns=columns)
    torque_signalNoise = pd.DataFrame(columns=columns)
    MAX_AMP = pd.DataFrame(columns=columns)
    fig, (x1, x2, xf1) = plt.subplots(3)
    fig, (x3, x4, xf2) = plt.subplots(3)
    fig, (x5, x6, xf3) = plt.subplots(3)
    fig, (x7, x8, xf4) = plt.subplots(3)
    fig, (x9, x10, xf5) = plt.subplots(3)
    fig, (x11, x12, xf6) = plt.subplots(3)
    
    i = 0
    for loop_file in os.listdir(File_path):
        route = os.path.join(File_path,loop_file)

        with ZipFile(route) as File:
            for file in File.namelist():
                with File.open(file) as f:
                    header = read_header(f)
                    #Set the index
                    date = header['date_time']
    #                 i = date.date()
                    #Read data
                    data = read_data(f)
                    mode = OperationMode(axis = header['axis'], mech_unit = header['mech_unit'], tool = header['tool'], rob_ax = header['rob_ax'], ext_ax = header['ext_ax'], prog_version = header['prog_version'])    
                    series = MccTimeSeries(data, header['date_time'], header.get('sampling_period', SAMPLING_PERIOD))
                    index = series.time_index
                    if ax1 in file:
                    # Extract features for axis1
                        d1 = standardized_torque(series.torque,mean[ax1], stv[ax1])
                        s1 = series.velocity
                        xf1.plot(s1)
                        
                        #ff_d1 = series.torque_ff
                        #xf1.plot(ff_d1)
                        xf1.set_xlabel('Timestamp')  
                        xf1.set_ylabel('Speed axis1')
                        d_m1, ref_m1,offset1 = cross_correlation(ref[ax1], d1) 
                        corr_1 = np.corrcoef(d_m1, ref_m1)
                        
                        if abs(offset1) > 10 or corr_1[0,1] <0.9:
                            continue
                        else:
                            torque_areaUnderSignal.loc[i, 'time'] = date.date()
                            torque_areaUnderSignal.loc[i, 'axis1'] = abs(np.trapz(abs(d_m1), dx = index[1] - index[0]) - np.trapz(abs(ref_m1), dx = index[1] - index[0]))
                            
                            torque_trimmedMean.loc[i, 'time'] = date.date()
                            torque_trimmedMean.loc[i, 'axis1'] = abs(trimmedMean(d_m1)- trimmedMean(ref_m1))

                            torque_signalNoise.loc[i, 'time'] = date.date()
                            torque_signalNoise.loc[i, 'axis1'] = abs((d_m1-ref_m1).mean()/(d_m1-ref_m1).std(ddof=0))
                            
                            RMS_Value.loc[i, 'time'] = date.date()
                            RMS_Value.loc[i, 'axis1'] = feature.root_mean_square(d_m1-ref_m1)

                            Skewness_value.loc[i, 'time'] = date.date()
                            Skewness_value.loc[i, 'axis1'] = feature.skewness(d_m1-ref_m1)

                            Kurtosis_value.loc[i, 'time'] = date.date()
                            Kurtosis_value.loc[i, 'axis1'] = feature.kurtosis(d_m1-ref_m1)

                            Energy.loc[i, 'time'] = date.date()
                            Energy.loc[i, 'axis1'] = feature.abs_energy(d_m1-ref_m1)

                            Stv.loc[i, 'time'] = date.date()
                            Stv.loc[i, 'axis1'] = (d_m1-ref_m1).std(ddof=0)

                            Variance.loc[i, 'time'] = date.date()
                            Variance.loc[i, 'axis1'] = (d_m1-ref_m1).var()

                            Impulse_factor.loc[i, 'time'] = date.date()
                            Impulse_factor.loc[i, 'axis1'] = max(d_m1-ref_m1)/abs(feature.mean(d_m1-ref_m1))

                            Shape_factor.loc[i, 'time'] = date.date()
                            Shape_factor.loc[i, 'axis1'] = feature.root_mean_square(d_m1-ref_m1)/abs(feature.mean(d_m1-ref_m1))
                            
                            MAX_AMP.loc[i, 'time'] = date.date()
                            MAX_AMP.loc[i, 'axis1'] = torque_fftFeatures_maxamplitude(d_m1-ref_m1)
                            
                            

                            x1.plot(d1)
                            x2.plot(d_m1)
                            x1.set_xlabel('Timestamp')  
                            x1.set_ylabel('Torque axis1')
                            x2.set_xlabel('Timestamp')  
                            x2.set_ylabel('Torque axis1')


                    elif ax2 in file:
                    # Extract features for axis2
                        ff_d2 = series.torque_ff
                        xf2.plot(ff_d2)
                        xf2.set_xlabel('Timestamp')  
                        xf2.set_ylabel('FF Torque axis2')

                        d2 = standardized_torque(series.torque,mean[ax2], stv[ax2])
                        d_m2, ref_m2, offset2 = cross_correlation(ref[ax2], d2)
                        corr_2 = np.corrcoef(d_m2, ref_m2)
                        if abs(offset2) > 10 or corr_2[0,1] <0.9:
                            continue
                        else:
                            x3.plot(d2)
                            x4.plot(d_m2)
                            x3.set_xlabel('Timestamp')  
                            x3.set_ylabel('Torque axis2')
                            x4.set_xlabel('Timestamp')  
                            x4.set_ylabel('Torque axis2')
                            torque_areaUnderSignal.loc[i, 'axis2'] = abs(np.trapz(abs(d_m2), dx = index[1] - index[0]) - np.trapz(abs(ref_m2), dx = index[1] - index[0]))
                            torque_trimmedMean.loc[i, 'axis2'] = abs(trimmedMean(d_m2)- trimmedMean(ref_m2))
                            torque_signalNoise.loc[i, 'axis2'] = abs((d_m2-ref_m2).mean()/(d_m2-ref_m2).std(ddof=0))
                            RMS_Value.loc[i, 'axis2'] = feature.root_mean_square(d_m2-ref_m2)
                            Skewness_value.loc[i, 'axis2'] = feature.skewness(d_m2-ref_m2)
                            Kurtosis_value.loc[i, 'axis2'] = feature.kurtosis(d_m2-ref_m2)
                            Energy.loc[i, 'axis2'] = feature.abs_energy(d_m2-ref_m2)
                            Stv.loc[i, 'axis2'] = (d_m2-ref_m2).std(ddof=0)
                            Variance.loc[i, 'axis2'] = (d_m2-ref_m2).var()
                            Impulse_factor.loc[i, 'axis2'] = max(d_m2-ref_m2)/abs(feature.mean(d_m2-ref_m2))
                            Shape_factor.loc[i, 'axis2'] = feature.root_mean_square(d_m2-ref_m2)/abs(feature.mean(d_m2-ref_m2))
                            MAX_AMP.loc[i, 'axis2'] = torque_fftFeatures_maxamplitude(d_m2-ref_m2)                        
                            
                            
                            
                    elif ax3 in file:
                    # Extract features for axis3
                        ff_d3 = series.torque_ff
                        xf3.plot(ff_d3)
                        xf3.set_xlabel('Timestamp')  
                        xf3.set_ylabel('FF Torque axis3')
                        d3 = standardized_torque(series.torque,mean[ax3], stv[ax3])
                        d_m3, ref_m3,offset3 = cross_correlation(ref[ax3], d3)
                        corr_3 = np.corrcoef(d_m3, ref_m3)
                        
                        if abs(offset3) > 10 or corr_3[0,1] <0.9:
                            continue
                        else:
                            
                            x5.plot(d3)
                            x6.plot(d_m3)
                            x5.set_xlabel('Timestamp')  
                            x5.set_ylabel('Torque axis3')
                            x6.set_xlabel('Timestamp')  
                            x6.set_ylabel('Torque axis3')
                            torque_areaUnderSignal.loc[i, 'axis3'] = abs(np.trapz(abs(d_m3), dx = index[1] - index[0]) - np.trapz(abs(ref_m3), dx = index[1] - index[0]))
                            torque_trimmedMean.loc[i, 'axis3'] = abs(trimmedMean(d_m3)- trimmedMean(ref_m3))
                            torque_signalNoise.loc[i, 'axis3'] = abs((d_m3-ref_m3).mean()/(d_m3-ref_m3).std(ddof=0))
                            RMS_Value.loc[i, 'axis3'] = feature.root_mean_square(d_m3-ref_m3)
                            Skewness_value.loc[i, 'axis3'] = feature.skewness(d_m3-ref_m3)
                            Kurtosis_value.loc[i, 'axis3'] = feature.kurtosis(d_m3-ref_m3)
                            Stv.loc[i, 'axis3'] = (d_m3-ref_m3).std(ddof=0)
                            Variance.loc[i, 'axis3'] = (d_m3-ref_m3).var()
                            Energy.loc[i, 'axis3'] = feature.abs_energy(d_m3-ref_m3)
                            Impulse_factor.loc[i, 'axis3'] = max(d_m3-ref_m3)/abs(feature.mean(d_m3-ref_m3))
                            Shape_factor.loc[i, 'axis3'] = feature.root_mean_square(d_m3-ref_m3)/abs(feature.mean(d_m3-ref_m3))
                            MAX_AMP.loc[i, 'axis3'] = torque_fftFeatures_maxamplitude(d_m3-ref_m3)                        
                        
                    elif ax4 in file:
                    # Extract features for axis4
                        ff_d4 = series.torque_ff
                        xf4.plot(ff_d4)
                        xf4.set_xlabel('Timestamp')  
                        xf4.set_ylabel('FF Torque axis4')
                        d4 = standardized_torque(series.torque,mean[ax4], stv[ax4])
                        d_m4, ref_m4, offset4 = cross_correlation(ref[ax4], d4)
                        corr_4 = np.corrcoef(d_m4, ref_m4)
                        if abs(offset4) > 10 or corr_4[0,1] < 0.9:
                            continue
                        else:
                            x7.plot(d4)

                            x8.plot(d_m4)
                            x7.set_xlabel('Timestamp')  
                            x7.set_ylabel('Torque axis4')
                            x8.set_xlabel('Timestamp')  
                            x8.set_ylabel('Torque axis4')
                            torque_areaUnderSignal.loc[i, 'axis4'] = abs(np.trapz(abs(d_m4), dx = index[1] - index[0]) - np.trapz(abs(ref_m4), dx = index[1] - index[0]))
                            torque_trimmedMean.loc[i, 'axis4'] = abs(trimmedMean(d_m4)- trimmedMean(ref_m4))
                            torque_signalNoise.loc[i, 'axis4'] = abs((d_m4-ref_m4).mean()/(d_m4-ref_m4).std(ddof=0))
                            RMS_Value.loc[i, 'axis4'] = feature.root_mean_square(d_m4-ref_m4)
                            Skewness_value.loc[i, 'axis4'] = feature.skewness(d_m4-ref_m4)
                            Kurtosis_value.loc[i, 'axis4'] = feature.kurtosis(d_m4-ref_m4)
                            Energy.loc[i, 'axis4'] = feature.abs_energy(d_m4-ref_m4)
                            Stv.loc[i, 'axis4'] = (d_m4-ref_m4).std(ddof=0)
                            Variance.loc[i, 'axis4'] = (d_m4-ref_m4).var()
                            Impulse_factor.loc[i, 'axis4'] = max(d_m4-ref_m4)/abs(feature.mean(d_m4-ref_m4))
                            Shape_factor.loc[i, 'axis4'] = feature.root_mean_square(d_m4-ref_m4)/abs(feature.mean(d_m4-ref_m4))
                            MAX_AMP.loc[i, 'axis4'] = torque_fftFeatures_maxamplitude(d_m4-ref_m4) 
                            
                    elif ax5 in file:
                    # Extract features for axis5
                        ff_d5 = series.torque_ff
                        xf5.plot(ff_d5)
                        xf5.set_xlabel('Timestamp')  
                        xf5.set_ylabel('FF Torque axis5')
                        d5 = standardized_torque(series.torque,mean[ax5], stv[ax5])
                        d_m5, ref_m5,offset5 = cross_correlation(ref[ax5], d5)
                        corr_5 = np.corrcoef(d_m5, ref_m5)
                        if abs(offset5) > 10 or corr_5[0,1] <0.9:
                            continue
                        else:
                            x9.plot(d5)
                            x10.plot(d_m5)
                            x9.set_xlabel('Timestamp')  
                            x9.set_ylabel('Torque axis5')
                            x10.set_xlabel('Timestamp')  
                            x10.set_ylabel('Torque axis5')
                            torque_areaUnderSignal.loc[i, 'axis5'] = abs(np.trapz(abs(d_m5), dx = index[1] - index[0]) - np.trapz(abs(ref_m5), dx = index[1] - index[0]))
                            torque_trimmedMean.loc[i, 'axis5'] = abs(trimmedMean(d_m5)- trimmedMean(ref_m5))
                            torque_signalNoise.loc[i, 'axis5'] = abs((d_m5-ref_m5).mean()/(d_m5-ref_m5).std(ddof=0))
                            RMS_Value.loc[i, 'axis5'] = feature.root_mean_square(d_m5-ref_m5)
                            Skewness_value.loc[i, 'axis5'] = feature.skewness(d_m5-ref_m5)
                            Kurtosis_value.loc[i, 'axis5'] = feature.kurtosis(d_m5-ref_m5)
                            Energy.loc[i, 'axis5'] = feature.abs_energy(d_m5-ref_m5)
                            Stv.loc[i, 'axis5'] = (d_m5-ref_m5).std(ddof=0)
                            Variance.loc[i, 'axis5'] = (d_m5-ref_m5).var()
                            Impulse_factor.loc[i, 'axis5'] = max(d_m5-ref_m5)/abs(feature.mean(d_m5-ref_m5))
                            Shape_factor.loc[i, 'axis5'] = feature.root_mean_square(d_m5-ref_m5)/abs(feature.mean(d_m5-ref_m5))
                            MAX_AMP.loc[i, 'axis5'] = torque_fftFeatures_maxamplitude(d_m5-ref_m5)         
                            
                    elif ax6 in file:
                    # Extract features for axis6
                        ff_d6 = series.torque_ff
                        xf6.plot(ff_d6)
                        xf6.set_xlabel('Timestamp')  
                        xf6.set_ylabel('FF Torque axis6')
                        d6 = standardized_torque(series.torque,mean[ax6], stv[ax6])
                        d_m6, ref_m6,offset6 = cross_correlation(ref[ax6], d6)
                        corr_6 = np.corrcoef(d_m6, ref_m6)
                        
                        if abs(offset6) > 10 or corr_6[0,1] <0.9:
                            continue
                        else:
                            
                            x11.plot(d6)
                            x12.plot(d_m6)
                            x11.set_xlabel('Timestamp')  
                            x11.set_ylabel('Torque axis6')
                            x12.set_xlabel('Timestamp')  
                            x12.set_ylabel('Torque axis6')
                            torque_areaUnderSignal.loc[i, 'axis6'] = abs(np.trapz(abs(d_m6), dx = index[1] - index[0]) - np.trapz(abs(ref_m6), dx = index[1] - index[0])) 
                            torque_trimmedMean.loc[i, 'axis6'] = abs(trimmedMean(d_m6)- trimmedMean(ref_m6))
                            torque_signalNoise.loc[i, 'axis6'] = abs((d_m6-ref_m6).mean()/(d_m6-ref_m6).std(ddof=0))
                            RMS_Value.loc[i, 'axis6'] = feature.root_mean_square(d_m6-ref_m6)
                            Skewness_value.loc[i, 'axis6'] = feature.skewness(d_m6-ref_m6)
                            Kurtosis_value.loc[i, 'axis6'] = feature.kurtosis(d_m6-ref_m6)
                            Energy.loc[i, 'axis6'] = feature.abs_energy(d_m6-ref_m6)
                            Stv.loc[i, 'axis6'] = (d_m6-ref_m6).std(ddof=0)
                            Variance.loc[i, 'axis6'] = (d_m6-ref_m6).var()
                            Impulse_factor.loc[i, 'axis6'] = max(d_m6-ref_m6)/feature.mean(d_m6-ref_m6)
                            Shape_factor.loc[i, 'axis6'] = feature.root_mean_square(d_m6-ref_m6)/abs(feature.mean(d_m6-ref_m6))
                            MAX_AMP.loc[i, 'axis6'] = torque_fftFeatures_maxamplitude(d_m6-ref_m6)                        
                            
        i = i + 1
    return RMS_Value,Stv,Impulse_factor,Shape_factor,torque_areaUnderSignal, torque_trimmedMean, torque_signalNoise, MAX_AMP
               


# In[17]:


#Failure Case 2
file_path = 'C:/Users/semihua/Desktop/Mingjie/test data/FC2_7142240_66-51134/1343-0024-56_DIAGDATA_20140822_185018__MCC.zip'
ref_FC2, mean_FC2, stv_FC2 = process_file(file_path)

File_path = 'C:/Users/semihua/Desktop/Mingjie/test data/FC2_7142240_66-51134/MCCFiles'
RMS_Value_FC2,Stv_FC2,Impulse_factor_FC2,Shape_factor_FC2,torque_areaUnderSignal_FC2, torque_trimmedMean_FC2,torque_signalNoise_FC2, MAX_AMP_FC2 = generate_more_feature(File_path,ref_FC2, mean_FC2, stv_FC2)


# In[18]:


RMS_Value_FC2 = RMS_Value_FC2.dropna(axis=0,how='any')
Stv_FC2 = Stv_FC2.dropna(axis=0,how='any')
torque_areaUnderSignal_FC2 = torque_areaUnderSignal_FC2.dropna(axis=0,how='any')
torque_trimmedMean_FC2 = torque_trimmedMean_FC2.dropna(axis=0,how='any')
torque_signalNoise_FC2 = torque_signalNoise_FC2.dropna(axis=0,how='any')
MAX_AMP_FC2 = MAX_AMP_FC2.dropna(axis=0,how='any')


# In[19]:


RMS_Value_FC2 = RMS_Value_FC2.groupby(by='time').mean()
Stv_FC2 = Stv_FC2.groupby(by='time').mean()
torque_areaUnderSignal_FC2 = torque_areaUnderSignal_FC2.groupby(by='time').mean()
torque_trimmedMean_FC2 = torque_trimmedMean_FC2.groupby(by='time').mean()
torque_signalNoise_FC2 = torque_signalNoise_FC2.groupby(by='time').mean()
MAX_AMP_FC2 = MAX_AMP_FC2.groupby(by='time').mean()


# In[20]:



torque_areaUnderSignal_FC2[0:50]


# In[21]:



RMS_Value_FC2.plot(figsize=(18,10))


# In[22]:


torque_signalNoise_FC2.plot(figsize=(18,10))


# In[23]:


#Failure Case 10
file_path = 'C:/Users/semihua/Desktop/Mingjie/test data/FC10_68363858_04-64139/04-64139_DIAGDATA_20170918_135452_3ece932f-a030-4c95-bd2c-e6d899ed8871_MCC.zip'
ref_FC10, mean_FC10, stv_FC10 = process_file(file_path)
File_path = 'C:/Users/semihua/Desktop/Mingjie/test data/FC10_68363858_04-64139/MCC Flies'
RMS_Value_FC10,Stv_FC10,Impulse_factor_FC10,Shape_factor_FC10, torque_areaUnderSignal_FC10, torque_trimmedMean_FC10,torque_signalNoise_FC10, MAX_AMP_FC10  = generate_more_feature(File_path,ref_FC10, mean_FC10, stv_FC10)


# In[24]:


RMS_Value_FC10 = RMS_Value_FC10.dropna(axis=0,how='any')
Stv_FC10 = Stv_FC10.dropna(axis=0,how='any')
torque_areaUnderSignal_FC10 = torque_areaUnderSignal_FC10.dropna(axis=0,how='any')
torque_trimmedMean_FC10 = torque_trimmedMean_FC10.dropna(axis=0,how='any')
torque_signalNoise_FC10 = torque_signalNoise_FC10.dropna(axis=0,how='any')
MAX_AMP_FC10 = MAX_AMP_FC10.dropna(axis=0,how='any')


# In[25]:


RMS_Value_FC10 = RMS_Value_FC10.groupby(by='time').mean()
Stv_FC10 = Stv_FC10.groupby(by='time').mean()
torque_areaUnderSignal_FC10 = torque_areaUnderSignal_FC10.groupby(by='time').mean()
torque_trimmedMean_FC10 = torque_trimmedMean_FC10.groupby(by='time').mean()
torque_signalNoise_FC10 = torque_signalNoise_FC10.groupby(by='time').mean()
MAX_AMP_FC10 = MAX_AMP_FC10.groupby(by='time').mean()


# In[26]:


RMS_Value_FC10.plot(figsize=(18,10))


# In[27]:


def generate_feature_update_reference(Filepath,ref,mean,stv):    
    File_path = Filepath
    ax1,ax2,ax3,ax4,ax5,ax6 = 'axis1','axis2','axis3','axis4','axis5','axis6'
    columns=['axis1','axis2','axis3','axis4','axis5','axis6']
    RMS_Value=pd.DataFrame(columns=columns)
    Skewness_value = pd.DataFrame(columns=columns)
    Kurtosis_value = pd.DataFrame(columns=columns)
    MAX_AMP = pd.DataFrame(columns=columns)
    Energy = pd.DataFrame(columns=columns)
    Stv = pd.DataFrame(columns=columns)
    Variance = pd.DataFrame(columns=columns)
    Impulse_factor = pd.DataFrame(columns=columns)
    Shape_factor = pd.DataFrame(columns=columns)
    MAX = pd.DataFrame(columns=columns)
    torque_areaUnderSignal = pd.DataFrame(columns=columns)
    torque_trimmedMean = pd.DataFrame(columns=columns)
    torque_signalNoise = pd.DataFrame(columns=columns)
    fig, (x1, x2, xf1) = plt.subplots(3)
    fig, (x3, x4, xf2) = plt.subplots(3)
    fig, (x5, x6, xf3) = plt.subplots(3)
    fig, (x7, x8, xf4) = plt.subplots(3)
    fig, (x9, x10, xf5) = plt.subplots(3)
    fig, (x11, x12, xf6) = plt.subplots(3)
    n = True
    i = 0
    drop = 0
    for loop_file in os.listdir(File_path):
        route = os.path.join(File_path,loop_file)

        with ZipFile(route) as File:
            for file in File.namelist():
                with File.open(file) as f:
                    header = read_header(f)
                    #Set the index
                    date = header['date_time']
    #                 i = date.date()
                    #Read data
                    data = read_data(f)
                    mode = OperationMode(axis = header['axis'], mech_unit = header['mech_unit'], tool = header['tool'], rob_ax = header['rob_ax'], ext_ax = header['ext_ax'], prog_version = header['prog_version'])    
                    series = MccTimeSeries(data, header['date_time'], header.get('sampling_period', SAMPLING_PERIOD))
                    index = series.time_index
                    if ax1 in file:
                    # Extract features for axis1
                        d1 = standardized_torque(series.torque,mean[ax1], stv[ax1])
                        s1 = series.velocity
                        xf1.plot(s1)
                        
                        #ff_d1 = series.torque_ff
                        #xf1.plot(ff_d1)
                        xf1.set_xlabel('Timestamp')  
                        xf1.set_ylabel('Speed axis1')
                        d_m1, ref_m1,offset1 = cross_correlation(ref[ax1], d1) 
                        corr_1 = np.corrcoef(d_m1, ref_m1)
                        if abs(offset1) > 10 or corr_1[0,1] <0.99:
                            if n == True:
                                ref, mean, stv = process_file(route)
                                drop = i
                                n = False


                        torque_areaUnderSignal.loc[i, 'time'] = date.date()
                        torque_areaUnderSignal.loc[i, 'axis1'] = abs(np.trapz(abs(d_m1), dx = index[1] - index[0]) - np.trapz(abs(ref_m1), dx = index[1] - index[0]))

                        torque_trimmedMean.loc[i, 'time'] = date.date()
                        torque_trimmedMean.loc[i, 'axis1'] = abs(trimmedMean(d_m1)- trimmedMean(ref_m1))

                        torque_signalNoise.loc[i, 'time'] = date.date()
                        torque_signalNoise.loc[i, 'axis1'] = (d_m1-ref_m1).mean()/(d_m1-ref_m1).std(ddof=0)

                        RMS_Value.loc[i, 'time'] = date.date()
                        RMS_Value.loc[i, 'axis1'] = feature.root_mean_square(d_m1-ref_m1)

                        Skewness_value.loc[i, 'time'] = date.date()
                        Skewness_value.loc[i, 'axis1'] = feature.skewness(d_m1-ref_m1)

                        Kurtosis_value.loc[i, 'time'] = date.date()
                        Kurtosis_value.loc[i, 'axis1'] = feature.kurtosis(d_m1-ref_m1)

                        Energy.loc[i, 'time'] = date.date()
                        Energy.loc[i, 'axis1'] = feature.abs_energy(d_m1-ref_m1)

                        Stv.loc[i, 'time'] = date.date()
                        Stv.loc[i, 'axis1'] = (d_m1-ref_m1).std(ddof=0)

                        Variance.loc[i, 'time'] = date.date()
                        Variance.loc[i, 'axis1'] = (d_m1-ref_m1).var()

                        Impulse_factor.loc[i, 'time'] = date.date()
                        Impulse_factor.loc[i, 'axis1'] = max(d_m1-ref_m1)/abs(feature.mean(d_m1-ref_m1))

                        Shape_factor.loc[i, 'time'] = date.date()
                        Shape_factor.loc[i, 'axis1'] = feature.root_mean_square(d_m1-ref_m1)/abs(feature.mean(d_m1-ref_m1))
                        
                        MAX_AMP.loc[i, 'time'] = date.date()
                        MAX_AMP.loc[i, 'axis1'] = torque_fftFeatures_maxamplitude(d_m1-ref_m1)

                        x1.plot(d1)
                        x2.plot(d_m1)
                        x1.set_xlabel('Timestamp')  
                        x1.set_ylabel('Torque axis1')
                        x2.set_xlabel('Timestamp')  
                        x2.set_ylabel('Torque axis1')


                    elif ax2 in file:
                    # Extract features for axis2
                        ff_d2 = series.torque_ff
                        xf2.plot(ff_d2)
                        xf2.set_xlabel('Timestamp')  
                        xf2.set_ylabel('FF Torque axis2')

                        d2 = standardized_torque(series.torque,mean[ax2], stv[ax2])
                        d_m2, ref_m2, offset2 = cross_correlation(ref[ax2], d2)
                        corr_2 = np.corrcoef(d_m2, ref_m2)
                        
                        if abs(offset2) > 10 or corr_2[0,1] <0.99:
                            if n == True:
                                ref, mean, stv = process_file(route)
                                drop = i
                                n = False
                        else:

                            x3.plot(d2)
                            x4.plot(d_m2)
                            x3.set_xlabel('Timestamp')  
                            x3.set_ylabel('Torque axis2')
                            x4.set_xlabel('Timestamp')  
                            x4.set_ylabel('Torque axis2')
                            torque_areaUnderSignal.loc[i, 'axis2'] = abs(np.trapz(abs(d_m2), dx = index[1] - index[0]) - np.trapz(abs(ref_m2), dx = index[1] - index[0]))
                            torque_trimmedMean.loc[i, 'axis2'] = abs(trimmedMean(d_m2)- trimmedMean(ref_m2))
                            torque_signalNoise.loc[i, 'axis2'] = (d_m2-ref_m2).mean()/(d_m2-ref_m2).std(ddof=0)
                            RMS_Value.loc[i, 'axis2'] = feature.root_mean_square(d_m2-ref_m2)
                            Skewness_value.loc[i, 'axis2'] = feature.skewness(d_m2-ref_m2)
                            Kurtosis_value.loc[i, 'axis2'] = feature.kurtosis(d_m2-ref_m2)
                            Energy.loc[i, 'axis2'] = feature.abs_energy(d_m2-ref_m2)
                            Stv.loc[i, 'axis2'] = (d_m2-ref_m2).std(ddof=0)
                            Variance.loc[i, 'axis2'] = (d_m2-ref_m2).var()
                            Impulse_factor.loc[i, 'axis2'] = max(d_m2-ref_m2)/abs(feature.mean(d_m2-ref_m2))
                            Shape_factor.loc[i, 'axis2'] = feature.root_mean_square(d_m2-ref_m2)/abs(feature.mean(d_m2-ref_m2))
                            MAX_AMP.loc[i, 'axis2'] = torque_fftFeatures_maxamplitude(d_m2-ref_m2)
                            
                    elif ax3 in file:
                    # Extract features for axis3
                        ff_d3 = series.torque_ff
                        xf3.plot(ff_d3)
                        xf3.set_xlabel('Timestamp')  
                        xf3.set_ylabel('FF Torque axis3')
                        d3 = standardized_torque(series.torque,mean[ax3], stv[ax3])
                        d_m3, ref_m3,offset3 = cross_correlation(ref[ax3], d3)
                        corr_3 = np.corrcoef(d_m3, ref_m3)
                        
                        if abs(offset3) > 10 or corr_3[0,1] <0.99:
                            if n == True:
                                ref, mean, stv = process_file(route)
                                drop = i
                                n = False
                        else:
                            
                            x5.plot(d3)
                            x6.plot(d_m3)
                            x5.set_xlabel('Timestamp')  
                            x5.set_ylabel('Torque axis3')
                            x6.set_xlabel('Timestamp')  
                            x6.set_ylabel('Torque axis3')
                            torque_areaUnderSignal.loc[i, 'axis3'] = abs(np.trapz(abs(d_m3), dx = index[1] - index[0]) - np.trapz(abs(ref_m3), dx = index[1] - index[0]))
                            torque_trimmedMean.loc[i, 'axis3'] = abs(trimmedMean(d_m3)- trimmedMean(ref_m3))
                            torque_signalNoise.loc[i, 'axis3'] = (d_m3-ref_m3).mean()/(d_m3-ref_m3).std(ddof=0)
                            RMS_Value.loc[i, 'axis3'] = feature.root_mean_square(d_m3-ref_m3)
                            Skewness_value.loc[i, 'axis3'] = feature.skewness(d_m3-ref_m3)
                            Kurtosis_value.loc[i, 'axis3'] = feature.kurtosis(d_m3-ref_m3)
                            Stv.loc[i, 'axis3'] = (d_m3-ref_m3).std(ddof=0)
                            Variance.loc[i, 'axis3'] = (d_m3-ref_m3).var()
                            Energy.loc[i, 'axis3'] = feature.abs_energy(d_m3-ref_m3)
                            Impulse_factor.loc[i, 'axis3'] = max(d_m3-ref_m3)/abs(feature.mean(d_m3-ref_m3))
                            Shape_factor.loc[i, 'axis3'] = feature.root_mean_square(d_m3-ref_m3)/abs(feature.mean(d_m3-ref_m3))
                            MAX_AMP.loc[i, 'axis3'] = torque_fftFeatures_maxamplitude(d_m3-ref_m3)
                    elif ax4 in file:
                    # Extract features for axis4
                        ff_d4 = series.torque_ff
                        xf4.plot(ff_d4)
                        xf4.set_xlabel('Timestamp')  
                        xf4.set_ylabel('FF Torque axis4')
                        d4 = standardized_torque(series.torque,mean[ax4], stv[ax4])
                        d_m4, ref_m4, offset4 = cross_correlation(ref[ax4], d4)
                        corr_4 = np.corrcoef(d_m4, ref_m4)
                        if abs(offset4) > 10 or corr_4[0,1] <0.99:
                            if n == True:
                                ref, mean, stv = process_file(route)
                                drop = i
                                n = False
                        else:
                            x7.plot(d4)

                            x8.plot(d_m4)
                            x7.set_xlabel('Timestamp')  
                            x7.set_ylabel('Torque axis4')
                            x8.set_xlabel('Timestamp')  
                            x8.set_ylabel('Torque axis4')
                            torque_areaUnderSignal.loc[i, 'axis4'] = abs(np.trapz(abs(d_m4), dx = index[1] - index[0]) - np.trapz(abs(ref_m4), dx = index[1] - index[0]))
                            torque_trimmedMean.loc[i, 'axis4'] = abs(trimmedMean(d_m4)- trimmedMean(ref_m4))
                            torque_signalNoise.loc[i, 'axis4'] = (d_m4-ref_m4).mean()/(d_m4-ref_m4).std(ddof=0)
                            RMS_Value.loc[i, 'axis4'] = feature.root_mean_square(d_m4-ref_m4)
                            Skewness_value.loc[i, 'axis4'] = feature.skewness(d_m4-ref_m4)
                            Kurtosis_value.loc[i, 'axis4'] = feature.kurtosis(d_m4-ref_m4)
                            Energy.loc[i, 'axis4'] = feature.abs_energy(d_m4-ref_m4)
                            Stv.loc[i, 'axis4'] = (d_m4-ref_m4).std(ddof=0)
                            Variance.loc[i, 'axis4'] = (d_m4-ref_m4).var()
                            Impulse_factor.loc[i, 'axis4'] = max(d_m4-ref_m4)/abs(feature.mean(d_m4-ref_m4))
                            Shape_factor.loc[i, 'axis4'] = feature.root_mean_square(d_m4-ref_m4)/abs(feature.mean(d_m4-ref_m4))
                            MAX_AMP.loc[i, 'axis4'] = torque_fftFeatures_maxamplitude(d_m4-ref_m4)
                    elif ax5 in file:
                    # Extract features for axis5
                        ff_d5 = series.torque_ff
                        xf5.plot(ff_d5)
                        xf5.set_xlabel('Timestamp')  
                        xf5.set_ylabel('FF Torque axis5')
                        d5 = standardized_torque(series.torque,mean[ax5], stv[ax5])
                        d_m5, ref_m5,offset5 = cross_correlation(ref[ax5], d5)
                        corr_5 = np.corrcoef(d_m5, ref_m5)
                        if abs(offset5) > 10 or corr_5[0,1] <0.99:
                            if n == True:
                                ref, mean, stv = process_file(route)
                                drop = i
                                n = False
                        else:
                            x9.plot(d5)
                            x10.plot(d_m5)
                            x9.set_xlabel('Timestamp')  
                            x9.set_ylabel('Torque axis5')
                            x10.set_xlabel('Timestamp')  
                            x10.set_ylabel('Torque axis5')
                            torque_areaUnderSignal.loc[i, 'axis5'] = abs(np.trapz(abs(d_m5), dx = index[1] - index[0]) - np.trapz(abs(ref_m5), dx = index[1] - index[0]))
                            torque_trimmedMean.loc[i, 'axis5'] = abs(trimmedMean(d_m5)- trimmedMean(ref_m5))
                            torque_signalNoise.loc[i, 'axis5'] = (d_m5-ref_m5).mean()/(d_m5-ref_m5).std(ddof=0)
                            RMS_Value.loc[i, 'axis5'] = feature.root_mean_square(d_m5-ref_m5)
                            Skewness_value.loc[i, 'axis5'] = feature.skewness(d_m5-ref_m5)
                            Kurtosis_value.loc[i, 'axis5'] = feature.kurtosis(d_m5-ref_m5)
                            Energy.loc[i, 'axis5'] = feature.abs_energy(d_m5-ref_m5)
                            Stv.loc[i, 'axis5'] = (d_m5-ref_m5).std(ddof=0)
                            Variance.loc[i, 'axis5'] = (d_m5-ref_m5).var()
                            Impulse_factor.loc[i, 'axis5'] = max(d_m5-ref_m5)/abs(feature.mean(d_m5-ref_m5))
                            Shape_factor.loc[i, 'axis5'] = feature.root_mean_square(d_m5-ref_m5)/abs(feature.mean(d_m5-ref_m5))
                            MAX_AMP.loc[i, 'axis5'] = torque_fftFeatures_maxamplitude(d_m5-ref_m5)
                    elif ax6 in file:
                    # Extract features for axis6
                        ff_d6 = series.torque_ff
                        xf6.plot(ff_d6)
                        xf6.set_xlabel('Timestamp')  
                        xf6.set_ylabel('FF Torque axis6')
                        d6 = standardized_torque(series.torque,mean[ax6], stv[ax6])
                        d_m6, ref_m6,offset6 = cross_correlation(ref[ax6], d6)
                        corr_6 = np.corrcoef(d_m6, ref_m6)
                        
                        if abs(offset6) > 10 or corr_6[0,1] <0.99:
                            if n == True:
                                ref, mean, stv = process_file(route)
                                drop = i
                                n = False
                        else:
                            
                            x11.plot(d6)
                            x12.plot(d_m6)
                            x11.set_xlabel('Timestamp')  
                            x11.set_ylabel('Torque axis6')
                            x12.set_xlabel('Timestamp')  
                            x12.set_ylabel('Torque axis6')
                            torque_areaUnderSignal.loc[i, 'axis6'] = abs(np.trapz(abs(d_m6), dx = index[1] - index[0]) - np.trapz(abs(ref_m6), dx = index[1] - index[0])) 
                            torque_trimmedMean.loc[i, 'axis6'] = abs(trimmedMean(d_m6)- trimmedMean(ref_m6))
                            torque_signalNoise.loc[i, 'axis6'] = (d_m6-ref_m6).mean()/(d_m6-ref_m6).std(ddof=0)
                            RMS_Value.loc[i, 'axis6'] = feature.root_mean_square(d_m6-ref_m6)
                            Skewness_value.loc[i, 'axis6'] = feature.skewness(d_m6-ref_m6)
                            Kurtosis_value.loc[i, 'axis6'] = feature.kurtosis(d_m6-ref_m6)
                            Energy.loc[i, 'axis6'] = feature.abs_energy(d_m6-ref_m6)
                            Stv.loc[i, 'axis6'] = (d_m6-ref_m6).std(ddof=0)
                            Variance.loc[i, 'axis6'] = (d_m6-ref_m6).var()
                            Impulse_factor.loc[i, 'axis6'] = max(d_m6-ref_m6)/feature.mean(d_m6-ref_m6)
                            Shape_factor.loc[i, 'axis6'] = feature.root_mean_square(d_m6-ref_m6)/abs(feature.mean(d_m6-ref_m6))
                            MAX_AMP.loc[i, 'axis6'] = torque_fftFeatures_maxamplitude(d_m6-ref_m6)
        i = i + 1
    return RMS_Value,Stv,Impulse_factor,Shape_factor,torque_areaUnderSignal, torque_trimmedMean, torque_signalNoise, MAX_AMP
               


# In[28]:


def generate_feature_refresh_reference(Filepath,ref,mean,stv, refresh_date):    
    File_path = Filepath
    ax1,ax2,ax3,ax4,ax5,ax6 = 'axis1','axis2','axis3','axis4','axis5','axis6'
    columns=['axis1','axis2','axis3','axis4','axis5','axis6']
    RMS_Value=pd.DataFrame(columns=columns)
    Skewness_value = pd.DataFrame(columns=columns)
    Kurtosis_value = pd.DataFrame(columns=columns)
    
    Energy = pd.DataFrame(columns=columns)
    Stv = pd.DataFrame(columns=columns)
    Variance = pd.DataFrame(columns=columns)
    Impulse_factor = pd.DataFrame(columns=columns)
    Shape_factor = pd.DataFrame(columns=columns)
    MAX_AMP = pd.DataFrame(columns=columns)
    torque_areaUnderSignal = pd.DataFrame(columns=columns)
    torque_trimmedMean = pd.DataFrame(columns=columns)
    torque_signalNoise = pd.DataFrame(columns=columns)
    MAX_AMP = pd.DataFrame(columns=columns)
    NTEST = 0
    OFFSET2 = []
    CORR2 = []
    fig, x1 = plt.subplots()
    i = 0
    for loop_file in os.listdir(File_path):
        route = os.path.join(File_path,loop_file)

        with ZipFile(route) as File:
            for file in File.namelist():
                with File.open(file) as f:
                    header = read_header(f)
                    #Set the index
                    date = header['date_time']
                    DATE = date.date()
                    #Read data
                    data = read_data(f)
                    mode = OperationMode(axis = header['axis'], mech_unit = header['mech_unit'], tool = header['tool'], rob_ax = header['rob_ax'], ext_ax = header['ext_ax'], prog_version = header['prog_version'])    
                    series = MccTimeSeries(data, header['date_time'], header.get('sampling_period', SAMPLING_PERIOD))
                    index = series.time_index
                    if DATE == refresh_date:
                        ref, mean, stv = process_file(route)
                        NTEST = i
                        break


                    else:
                        if ax1 in file:
                        # Extract features for axis1
                            d1 = standardized_torque(series.torque,mean[ax1], stv[ax1])
                            d_m1, ref_m1,offset1 = cross_correlation(ref[ax1], d1) 
#                             OFFSET1.append(offset1)
                            corr_1 = np.corrcoef(d_m1, ref_m1)

                            if abs(offset1) > 10 or corr_1[0,1] <0.99:
                                continue
                            else:
                                torque_areaUnderSignal.loc[i, 'time'] = date.date()
                                torque_areaUnderSignal.loc[i, 'axis1'] = abs(np.trapz(abs(d_m1), dx = index[1] - index[0]) - np.trapz(abs(ref_m1), dx = index[1] - index[0]))

                                torque_trimmedMean.loc[i, 'time'] = date.date()
                                torque_trimmedMean.loc[i, 'axis1'] = abs(trimmedMean(d_m1)- trimmedMean(ref_m1))

                                torque_signalNoise.loc[i, 'time'] = date.date()
                                torque_signalNoise.loc[i, 'axis1'] = abs((d_m1-ref_m1).mean()/(d_m1-ref_m1).std(ddof=0))

                                RMS_Value.loc[i, 'time'] = date.date()
                                RMS_Value.loc[i, 'axis1'] = feature.root_mean_square(d_m1-ref_m1)

                                Skewness_value.loc[i, 'time'] = date.date()
                                Skewness_value.loc[i, 'axis1'] = feature.skewness(d_m1-ref_m1)

                                Kurtosis_value.loc[i, 'time'] = date.date()
                                Kurtosis_value.loc[i, 'axis1'] = feature.kurtosis(d_m1-ref_m1)

                                Energy.loc[i, 'time'] = date.date()
                                Energy.loc[i, 'axis1'] = feature.abs_energy(d_m1-ref_m1)

                                Stv.loc[i, 'time'] = date.date()
                                Stv.loc[i, 'axis1'] = (d_m1-ref_m1).std(ddof=0)

                                Variance.loc[i, 'time'] = date.date()
                                Variance.loc[i, 'axis1'] = (d_m1-ref_m1).var()

                                Impulse_factor.loc[i, 'time'] = date.date()
                                Impulse_factor.loc[i, 'axis1'] = max(d_m1-ref_m1)/abs(feature.mean(d_m1-ref_m1))

                                Shape_factor.loc[i, 'time'] = date.date()
                                Shape_factor.loc[i, 'axis1'] = feature.root_mean_square(d_m1-ref_m1)/abs(feature.mean(d_m1-ref_m1))

                                MAX_AMP.loc[i, 'time'] = date.date()
                                MAX_AMP.loc[i, 'axis1'] = torque_fftFeatures_maxamplitude(d_m1-ref_m1)


                        elif ax2 in file:
                        # Extract features for axis2

                            d2 = standardized_torque(series.torque,mean[ax2], stv[ax2])
                            d_m2, ref_m2, offset2 = cross_correlation(ref[ax2], d2)
                            OFFSET2.append(offset2)
                            corr_2 = np.corrcoef(d_m2, ref_m2)
                            
                            if abs(offset2) > 10 or corr_2[0,1] <0.99:
                                continue
                            else:
    
                                torque_areaUnderSignal.loc[i, 'axis2'] = abs(np.trapz(abs(d_m2), axis = -1, dx = index[1] - index[0]) - np.trapz(abs(ref_m2), axis = -1, dx = index[1] - index[0]))
                                torque_trimmedMean.loc[i, 'axis2'] = abs(trimmedMean(d_m2)- trimmedMean(ref_m2))
                                torque_signalNoise.loc[i, 'axis2'] = abs((d_m2-ref_m2).mean()/(d_m2-ref_m2).std(ddof=0))
                                RMS_Value.loc[i, 'axis2'] = feature.root_mean_square(d_m2-ref_m2)
                                Skewness_value.loc[i, 'axis2'] = feature.skewness(d_m2-ref_m2)
                                Kurtosis_value.loc[i, 'axis2'] = feature.kurtosis(d_m2-ref_m2)
                                Energy.loc[i, 'axis2'] = feature.abs_energy(d_m2-ref_m2)
                                Stv.loc[i, 'axis2'] = (d_m2-ref_m2).std(ddof=0)
                                Variance.loc[i, 'axis2'] = (d_m2-ref_m2).var()
                                Impulse_factor.loc[i, 'axis2'] = max(d_m2-ref_m2)/abs(feature.mean(d_m2-ref_m2))
                                Shape_factor.loc[i, 'axis2'] = feature.root_mean_square(d_m2-ref_m2)/abs(feature.mean(d_m2-ref_m2))
                                MAX_AMP.loc[i, 'axis2'] = torque_fftFeatures_maxamplitude(d_m2-ref_m2)
                                if i >200 and i < 210:
                                    x1.plot(ref_m2, 'b')
                                    x1.plot(d_m2, 'g')
                                    x1.set_xlabel('Timestamp')  
                                    x1.set_ylabel('Torque axis4')
                                


                        elif ax3 in file:
                        # Extract features for axis3
                            d3 = standardized_torque(series.torque,mean[ax3], stv[ax3])
                            d_m3, ref_m3,offset3 = cross_correlation(ref[ax3], d3)
                            corr_3 = np.corrcoef(d_m3, ref_m3)

                            if abs(offset3) > 10 or corr_3[0,1] <0.99:
                                continue
                            else:
                                torque_areaUnderSignal.loc[i, 'axis3'] = abs(np.trapz(abs(d_m3), dx = index[1] - index[0]) - np.trapz(abs(ref_m3), dx = index[1] - index[0]))
                                torque_trimmedMean.loc[i, 'axis3'] = abs(trimmedMean(d_m3)- trimmedMean(ref_m3))
                                torque_signalNoise.loc[i, 'axis3'] = abs((d_m3-ref_m3).mean()/(d_m3-ref_m3).std(ddof=0))
                                RMS_Value.loc[i, 'axis3'] = feature.root_mean_square(d_m3-ref_m3)
                                Skewness_value.loc[i, 'axis3'] = feature.skewness(d_m3-ref_m3)
                                Kurtosis_value.loc[i, 'axis3'] = feature.kurtosis(d_m3-ref_m3)
                                Stv.loc[i, 'axis3'] = (d_m3-ref_m3).std(ddof=0)
                                Variance.loc[i, 'axis3'] = (d_m3-ref_m3).var()
                                Energy.loc[i, 'axis3'] = feature.abs_energy(d_m3-ref_m3)
                                Impulse_factor.loc[i, 'axis3'] = max(d_m3-ref_m3)/abs(feature.mean(d_m3-ref_m3))
                                Shape_factor.loc[i, 'axis3'] = feature.root_mean_square(d_m3-ref_m3)/abs(feature.mean(d_m3-ref_m3))
                                MAX_AMP.loc[i, 'axis3'] = torque_fftFeatures_maxamplitude(d_m3-ref_m3)                        

                        elif ax4 in file:
                        # Extract features for axis4
                            d4 = standardized_torque(series.torque,mean[ax4], stv[ax4])
                            d_m4, ref_m4, offset4 = cross_correlation(ref[ax4], d4)
                            corr_4 = np.corrcoef(d_m4, ref_m4)
                            if abs(offset4) > 10 or corr_4[0,1] < 0.99:
                                continue
                            else:
                                torque_areaUnderSignal.loc[i, 'axis4'] = abs(np.trapz(abs(d_m4), dx = index[1] - index[0]) - np.trapz(abs(ref_m4), dx = index[1] - index[0]))
                                torque_trimmedMean.loc[i, 'axis4'] = abs(trimmedMean(d_m4)- trimmedMean(ref_m4))
                                torque_signalNoise.loc[i, 'axis4'] = abs((d_m4-ref_m4).mean()/(d_m4-ref_m4).std(ddof=0))
                                RMS_Value.loc[i, 'axis4'] = feature.root_mean_square(d_m4-ref_m4)
                                Skewness_value.loc[i, 'axis4'] = feature.skewness(d_m4-ref_m4)
                                Kurtosis_value.loc[i, 'axis4'] = feature.kurtosis(d_m4-ref_m4)
                                Energy.loc[i, 'axis4'] = feature.abs_energy(d_m4-ref_m4)
                                Stv.loc[i, 'axis4'] = (d_m4-ref_m4).std(ddof=0)
                                Variance.loc[i, 'axis4'] = (d_m4-ref_m4).var()
                                Impulse_factor.loc[i, 'axis4'] = max(d_m4-ref_m4)/abs(feature.mean(d_m4-ref_m4))
                                Shape_factor.loc[i, 'axis4'] = feature.root_mean_square(d_m4-ref_m4)/abs(feature.mean(d_m4-ref_m4))
                                MAX_AMP.loc[i, 'axis4'] = torque_fftFeatures_maxamplitude(d_m4-ref_m4) 

                        elif ax5 in file:
                        # Extract features for axis5
                            d5 = standardized_torque(series.torque,mean[ax5], stv[ax5])
                            d_m5, ref_m5,offset5 = cross_correlation(ref[ax5], d5)
                            corr_5 = np.corrcoef(d_m5, ref_m5)
                            CORR2.append(corr_5)
                            if abs(offset5) > 10 or corr_5[0,1] <0.99:
                                continue
                            else:
                                torque_areaUnderSignal.loc[i, 'axis5'] = abs(np.trapz(abs(d_m5), dx = index[1] - index[0]) - np.trapz(abs(ref_m5), dx = index[1] - index[0]))
                                torque_trimmedMean.loc[i, 'axis5'] = abs(trimmedMean(d_m5)- trimmedMean(ref_m5))
                                torque_signalNoise.loc[i, 'axis5'] = abs((d_m5-ref_m5).mean()/(d_m5-ref_m5).std(ddof=0))
                                RMS_Value.loc[i, 'axis5'] = feature.root_mean_square(d_m5-ref_m5)
                                Skewness_value.loc[i, 'axis5'] = feature.skewness(d_m5-ref_m5)
                                Kurtosis_value.loc[i, 'axis5'] = feature.kurtosis(d_m5-ref_m5)
                                Energy.loc[i, 'axis5'] = feature.abs_energy(d_m5-ref_m5)
                                Stv.loc[i, 'axis5'] = (d_m5-ref_m5).std(ddof=0)
                                Variance.loc[i, 'axis5'] = (d_m5-ref_m5).var()
                                Impulse_factor.loc[i, 'axis5'] = max(d_m5-ref_m5)/abs(feature.mean(d_m5-ref_m5))
                                Shape_factor.loc[i, 'axis5'] = feature.root_mean_square(d_m5-ref_m5)/abs(feature.mean(d_m5-ref_m5))
                                MAX_AMP.loc[i, 'axis5'] = torque_fftFeatures_maxamplitude(d_m5-ref_m5)         

                        elif ax6 in file:
                        # Extract features for axis6
                            d6 = standardized_torque(series.torque,mean[ax6], stv[ax6])
                            d_m6, ref_m6,offset6 = cross_correlation(ref[ax6], d6)
                            corr_6 = np.corrcoef(d_m6, ref_m6)

                            if abs(offset6) > 10 or corr_6[0,1] <0.99:
                                continue
                            else:
                                torque_areaUnderSignal.loc[i, 'axis6'] = abs(np.trapz(abs(d_m6), dx = index[1] - index[0]) - np.trapz(abs(ref_m6), dx = index[1] - index[0]))
                                torque_trimmedMean.loc[i, 'axis6'] = abs(trimmedMean(d_m6)- trimmedMean(ref_m6))
                                torque_signalNoise.loc[i, 'axis6'] = abs((d_m6-ref_m6).mean()/(d_m6-ref_m6).std(ddof=0))
                                RMS_Value.loc[i, 'axis6'] = feature.root_mean_square(d_m6-ref_m6)
                                Skewness_value.loc[i, 'axis6'] = feature.skewness(d_m6-ref_m6)
                                Kurtosis_value.loc[i, 'axis6'] = feature.kurtosis(d_m6-ref_m6)
                                Energy.loc[i, 'axis6'] = feature.abs_energy(d_m6-ref_m6)
                                Stv.loc[i, 'axis6'] = (d_m6-ref_m6).std(ddof=0)
                                Variance.loc[i, 'axis6'] = (d_m6-ref_m6).var()
                                Impulse_factor.loc[i, 'axis6'] = max(d_m6-ref_m6)/feature.mean(d_m6-ref_m6)
                                Shape_factor.loc[i, 'axis6'] = feature.root_mean_square(d_m6-ref_m6)/abs(feature.mean(d_m6-ref_m6))
                                MAX_AMP.loc[i, 'axis6'] = torque_fftFeatures_maxamplitude(d_m6-ref_m6)                        
                            
        i = i + 1
    return RMS_Value,Stv,Impulse_factor,Shape_factor,torque_areaUnderSignal, torque_trimmedMean, torque_signalNoise, MAX_AMP, NTEST, CORR2
               


# In[29]:


date_FC1 = date(2015, 12, 24)
#Failure Case 1

file_path = 'C:/Users/semihua/Desktop/Mingjie/test data/FC1_113727087_6700-101962/1426-0051-56_DIAGDATA_20151016_090000__MCC.zip'
ref_FC1, mean_FC1, stv_FC1 = process_file(file_path)

File_path = 'C:/Users/semihua/Desktop/Mingjie/test data/FC1_113727087_6700-101962/MCCFiles'
RMS_Value_FC1,Stv_FC1,Impulse_factor_FC1,Shape_factor_FC1,torque_areaUnderSignal_FC1, torque_trimmedMean_FC1,torque_signalNoise_FC1, MAX_AMP_FC1, drop_F1, CORR1 = generate_feature_refresh_reference(File_path,ref_FC1, mean_FC1, stv_FC1, date_FC1)


# In[30]:


RMS_Value_FC1 = RMS_Value_FC1.dropna(axis=0,how='any')
Stv_FC1 = Stv_FC1.dropna(axis=0,how='any')
torque_areaUnderSignal_FC1 = torque_areaUnderSignal_FC1.dropna(axis=0,how='any')
torque_trimmedMean_FC1 = torque_trimmedMean_FC1.dropna(axis=0,how='any')
torque_signalNoise_FC1 = torque_signalNoise_FC1.dropna(axis=0,how='any')
MAX_AMP_FC1 = MAX_AMP_FC1.dropna(axis=0,how='any')


# In[31]:


RMS_Value_FC1 = RMS_Value_FC1.groupby(by='time').mean()
Stv_FC1 = Stv_FC1.groupby(by='time').mean()
torque_areaUnderSignal_FC1 = torque_areaUnderSignal_FC1.groupby(by='time').mean()
torque_trimmedMean_FC1 = torque_trimmedMean_FC1.groupby(by='time').mean()
torque_signalNoise_FC1 = torque_signalNoise_FC1.groupby(by='time').mean()
MAX_AMP_FC1 = MAX_AMP_FC1.groupby(by='time').mean()


# In[32]:




RMS_Value_FC1.plot(figsize=(18,10), ylim =(0,0.4))


# In[33]:


#Failure Case 6
date_FC6 = date(2015, 8, 21)
file_path = 'C:/Users/semihua/Desktop/Mingjie/test data/FC6_33016615_66-62028/1218-0079-56_DIAGDATA_20150410_142226__MCC.zip'
ref_FC6, mean_FC6, stv_FC6 = process_file(file_path)

File_path = 'C:/Users/semihua/Desktop/Mingjie/test data/FC6_33016615_66-62028/MCCFiles'
RMS_Value_FC6,Stv_FC6,Impulse_factor_FC6,Shape_factor_FC6,torque_areaUnderSignal_FC6, torque_trimmedMean_FC6,torque_signalNoise_FC6, MAX_AMP_FC6, drop_FC6, CORR6 = generate_feature_refresh_reference(File_path,ref_FC6, mean_FC6, stv_FC6, date_FC6)


# In[34]:


CORR6[50:]


# In[35]:


#Failure Case 6
date_FC6 = date(2015, 9, 4)
file_path = 'C:/Users/semihua/Desktop/Mingjie/test data/FC6_33016615_66-62028/1218-0079-56_DIAGDATA_20150410_142226__MCC.zip'
ref_FC6, mean_FC6, stv_FC6 = process_file(file_path)

File_path = 'C:/Users/semihua/Desktop/Mingjie/test data/FC6_33016615_66-62028/MCCFiles'
RMS_Value_FC6,Stv_FC6,Impulse_factor_FC6,Shape_factor_FC6,torque_areaUnderSignal_FC6, torque_trimmedMean_FC6,torque_signalNoise_FC6, MAX_AMP_FC6 = generate_more_feature(File_path,ref_FC6, mean_FC6, stv_FC6)


# In[36]:


RMS_Value_FC6 = RMS_Value_FC6.dropna(axis=0,how='any')
Stv_FC6 = Stv_FC6.dropna(axis=0,how='any')
torque_areaUnderSignal_FC6 = torque_areaUnderSignal_FC6.dropna(axis=0,how='any')
torque_trimmedMean_FC6 = torque_trimmedMean_FC6.dropna(axis=0,how='any')
torque_signalNoise_FC6 = torque_signalNoise_FC6.dropna(axis=0,how='any')
MAX_AMP_FC6 = MAX_AMP_FC6.dropna(axis=0,how='any')


# In[37]:


RMS_Value_FC6 = RMS_Value_FC6.groupby(by='time').mean()
Stv_FC6 = Stv_FC6.groupby(by='time').mean()
torque_areaUnderSignal_FC6 = torque_areaUnderSignal_FC6.groupby(by='time').mean()
torque_trimmedMean_FC6 = torque_trimmedMean_FC6.groupby(by='time').mean()
torque_signalNoise_FC6 = torque_signalNoise_FC6.groupby(by='time').mean()
MAX_AMP_FC6 = MAX_AMP_FC6.groupby(by='time').mean()


# In[38]:


RMS_Value_FC6.plot(figsize=(18,10), ylim = (0,0.6))


# In[39]:


file_path = 'C:/Users/semihua/Desktop/Mingjie/test data/FC7_130270916_7600-102993/7600-102993_DIAGDATA_20160331_050019__MCC.zip'
ref_FC7, mean_FC7, stv_FC7 = process_file(file_path)
File_path = 'C:/Users/semihua/Desktop/Mingjie/test data/FC7_130270916_7600-102993/MCCFiles'
RMS_Value_FC7,Stv_FC7,Impulse_factor_FC7,Shape_factor_FC7, torque_areaUnderSignal_FC7, torque_trimmedMean_FC7,torque_signalNoise_FC7, MAX_AMP_FC7 = generate_more_feature(File_path,ref_FC7, mean_FC7, stv_FC7)


# In[40]:


date_FC7 = date(2016, 7, 3)
file_path = 'C:/Users/semihua/Desktop/Mingjie/test data/FC7_130270916_7600-102993/7600-102993_DIAGDATA_20160331_050019__MCC.zip'
ref_FC7, mean_FC7, stv_FC7 = process_file(file_path)
File_path = 'C:/Users/semihua/Desktop/Mingjie/test data/FC7_130270916_7600-102993/MCCFiles'
RMS_Value_FC7,Stv_FC7,Impulse_factor_FC7,Shape_factor_FC7, torque_areaUnderSignal_FC7, torque_trimmedMean_FC7,torque_signalNoise_FC7, MAX_AMP_FC7, drop_FC7, CORR7 = generate_feature_refresh_reference(File_path,ref_FC7, mean_FC7, stv_FC7, date_FC7)


# In[41]:


torque_areaUnderSignal_FC7[0:30]


# In[42]:


RMS_Value_FC7 = RMS_Value_FC7.dropna(axis=0,how='any')
Stv_FC7 = Stv_FC7.dropna(axis=0,how='any')
torque_areaUnderSignal_FC7 = torque_areaUnderSignal_FC7.dropna(axis=0,how='any')
torque_trimmedMean_FC7 = torque_trimmedMean_FC7.dropna(axis=0,how='any')
torque_signalNoise_FC7 = torque_signalNoise_FC7.dropna(axis=0,how='any')
MAX_AMP_FC7 = MAX_AMP_FC7.dropna(axis=0,how='any')


# In[43]:


RMS_Value_FC7 = RMS_Value_FC7.groupby(by='time').mean()
Stv_FC7 = Stv_FC7.groupby(by='time').mean()
torque_areaUnderSignal_FC7 = torque_areaUnderSignal_FC7.groupby(by='time').mean()
torque_trimmedMean_FC7 = torque_trimmedMean_FC7.groupby(by='time').mean()
torque_signalNoise_FC7 = torque_signalNoise_FC7.groupby(by='time').mean()
MAX_AMP_FC7 = MAX_AMP_FC7.groupby(by='time').mean()


# In[44]:


torque_areaUnderSignal_FC7.plot(figsize=(18,10), ylim =(0,0.2))


# In[45]:


file_path = 'C:/Users/semihua/Desktop/Mingjie/test data/FC8_103507723_6700-100888/1237-0041-56_DIAGDATA_20141107_131229_1a2edf8e-8453-4210-8f9d-4bbc65e75273_MCC.zip'
ref_FC8, mean_FC8, stv_FC8 = process_file(file_path)
File_path = 'C:/Users/semihua/Desktop/Mingjie/test data/FC8_103507723_6700-100888/MCCFiles'
RMS_Value_FC8,Stv_FC8,Impulse_factor_FC8,Shape_factor_FC8, torque_areaUnderSignal_FC8, torque_trimmedMean_FC8,torque_signalNoise_FC8, MAX_AMP_FC8 = generate_more_feature(File_path,ref_FC8, mean_FC8, stv_FC8)


# In[46]:


RMS_Value_FC8[30:60]


# In[47]:


RMS_Value_FC8 = RMS_Value_FC8.dropna(axis=0,how='any')
Stv_FC8 = Stv_FC8.dropna(axis=0,how='any')
torque_areaUnderSignal_FC8 = torque_areaUnderSignal_FC8.dropna(axis=0,how='any')
torque_trimmedMean_FC8 = torque_trimmedMean_FC8.dropna(axis=0,how='any')
torque_signalNoise_FC8 = torque_signalNoise_FC8.dropna(axis=0,how='any')
MAX_AMP_FC8 = MAX_AMP_FC8.dropna(axis=0,how='any')


# In[48]:


torque_areaUnderSignal_FC8.plot(figsize=(18,10))


# In[49]:


RMS_Value_FC8 = RMS_Value_FC8.groupby(by='time').mean()
Stv_FC8 = Stv_FC8.groupby(by='time').mean()
torque_areaUnderSignal_FC8 = torque_areaUnderSignal_FC8.groupby(by='time').mean()
torque_trimmedMean_FC8 = torque_trimmedMean_FC8.groupby(by='time').mean()
torque_signalNoise_FC8 = torque_signalNoise_FC8.groupby(by='time').mean()
MAX_AMP_FC8 = MAX_AMP_FC8.groupby(by='time').mean()


# In[50]:


torque_areaUnderSignal_FC8.plot(figsize=(18,10))


# In[51]:


file_path = 'C:/Users/semihua/Desktop/Mingjie/test data/FC11_117360760_6640-108111/6640-108111_DIAGDATA_20160223_083249_3f0c9af8-e4b6-44de-ba75-53e13801feb1_MCC.zip'
ref_FC11, mean_FC11, stv_FC11 = process_file(file_path)
date_FC11 = date(2016, 4, 18)
File_path = 'C:/Users/semihua/Desktop/Mingjie/test data/FC11_117360760_6640-108111/MCCFiles'
RMS_Value_FC11,Stv_FC11,Impulse_factor_FC11,Shape_factor_FC11, torque_areaUnderSignal_FC11, torque_trimmedMean_FC11,torque_signalNoise_FC11, MAX_AMP_FC11,drop_FC11,CORR11 = generate_feature_refresh_reference(File_path,ref_FC11, mean_FC11, stv_FC11, date_FC11)


# In[52]:


drop_FC11


# In[53]:


RMS_Value_FC11 = RMS_Value_FC11.dropna(axis=0,how='any')
Stv_FC11 = Stv_FC11.dropna(axis=0,how='any')
torque_areaUnderSignal_FC11 = torque_areaUnderSignal_FC11.dropna(axis=0,how='any')
torque_trimmedMean_FC11 = torque_trimmedMean_FC11.dropna(axis=0,how='any')
torque_signalNoise_FC11 = torque_signalNoise_FC11.dropna(axis=0,how='any')
MAX_AMP_FC11 = MAX_AMP_FC11.dropna(axis=0,how='any')


# In[54]:


RMS_Value_FC11 = RMS_Value_FC11.groupby(by='time').mean()
Stv_FC11 = Stv_FC11.groupby(by='time').mean()
torque_areaUnderSignal_FC11 = torque_areaUnderSignal_FC11.groupby(by='time').mean()
torque_trimmedMean_FC11 = torque_trimmedMean_FC11.groupby(by='time').mean()
torque_signalNoise_FC11 = torque_signalNoise_FC11.groupby(by='time').mean()
MAX_AMP_FC11 = MAX_AMP_FC11.groupby(by='time').mean()


# In[55]:


RMS_Value_FC11.plot(figsize=(18,10))


# In[56]:


torque_areaUnderSignal_FC11.plot(figsize=(18,10))


# In[57]:


torque_areaUnderSignal_FC11[330:]


# In[58]:


file_path = 'C:/Users/semihua/Desktop/Mingjie/test data/FC12_130270916_7600-102993/7600-102993_DIAGDATA_20160331_050019__MCC.zip'
ref_FC12, mean_FC12, stv_FC12 = process_file(file_path)
date_FC12 = date(2017, 5, 19)
File_path = 'C:/Users/semihua/Desktop/Mingjie/test data/FC12_130270916_7600-102993/MCCFiles'
RMS_Value_FC12,Stv_FC12,Impulse_factor_FC12,Shape_factor_FC12, torque_areaUnderSignal_FC12, torque_trimmedMean_FC12,torque_signalNoise_FC12, MAX_AMP_FC12,drop_FC12, CORR2 = generate_feature_refresh_reference(File_path,ref_FC12, mean_FC12, stv_FC12,date_FC12)


# In[59]:


RMS_Value_FC12 = RMS_Value_FC12.dropna(axis=0,how='any')
Stv_FC12 = Stv_FC12.dropna(axis=0,how='any')
torque_areaUnderSignal_FC12 = torque_areaUnderSignal_FC12.dropna(axis=0,how='any')
torque_trimmedMean_FC12 = torque_trimmedMean_FC12.dropna(axis=0,how='any')
torque_signalNoise_FC12 = torque_signalNoise_FC12.dropna(axis=0,how='any')
MAX_AMP_FC12 = MAX_AMP_FC12.dropna(axis=0,how='any')


# In[60]:


RMS_Value_FC12 = RMS_Value_FC12.groupby(by='time').mean()
Stv_FC12 = Stv_FC12.groupby(by='time').mean()
torque_areaUnderSignal_FC12 = torque_areaUnderSignal_FC12.groupby(by='time').mean()
torque_trimmedMean_FC12 = torque_trimmedMean_FC12.groupby(by='time').mean()
torque_signalNoise_FC12 = torque_signalNoise_FC12.groupby(by='time').mean()
MAX_AMP_FC12 = MAX_AMP_FC12.groupby(by='time').mean()


# In[61]:


torque_areaUnderSignal_FC11[200:230]


# In[62]:


Stv_FC12.plot(figsize=(18,10))


# In[63]:


torque_areaUnderSignal_FC12.plot(figsize=(18,10))


# In[166]:


line_FC10 = [date(2019, 4, 18),date(2019, 6, 5)]
line_FC2 = [date(2015, 3, 20),date(2015, 4, 3)]
line_FC1 = [date(2016, 5, 6),date(2016, 5, 15)]
line_FC6 = [date(2016, 10, 1),date(2017, 2, 4)]
line_FC7 = [date(2017, 4, 20),date(2017, 4, 26)]
line_FC8 = [date(2016, 10, 19),date(2017, 1, 23)]
line_FC11 = [date(2017, 11, 26),date(2017, 12, 6)]
line_FC12 = [[date(2017, 12, 17),date(2017, 12, 22)],
            [date(2017,4,22), date(2017,4,27)]]
Failure_period = [line_FC10, line_FC2, line_FC1, line_FC6, line_FC7, line_FC8,line_FC11, line_FC12]


# In[65]:



axis_list = ['axis1','axis2','axis3','axis4','axis5','axis6']
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(18,10))

data = {
    'FC10 failure axis2': RMS_Value_FC10,
    'FC2 failure axis6': RMS_Value_FC2,
    'FC1 failure axis2': RMS_Value_FC1,
    'FC6 failure axis4': RMS_Value_FC6,
    'FC7 failure axis2': RMS_Value_FC7,
    'FC8 failure axis3': RMS_Value_FC8,
    'FC11 failure axis2': RMS_Value_FC11,
    'FC12 failure axis5': RMS_Value_FC12
}

for i, (fc, d) in enumerate(data.items()):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    for index in axis_list:
        ax.plot(d[index], label=index)
        ax.set_title(f'Failure Case {fc}')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('RMS_Value')
        ax.set_ylim(0, .5)
        ax.legend()
    if isinstance(Failure_period[i][0], list):
        for x in Failure_period[i]:
            ax.axvspan(x[0], x[1], alpha=0.2, color='red')
    else:
        ax.axvspan(Failure_period[i][0], Failure_period[i][1], alpha=0.2, color='red')
    ax.scatter(x = d.index, y =np.full(shape=len(d.index), fill_value=0.4),s=0.1, color='r', label= 'Existed Data Point' )
    ax.legend(loc= "best",fontsize="6")


plt.tight_layout()


# In[66]:


axis_list = ['axis1','axis2','axis3','axis4','axis5','axis6']
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(18,10))

data = {
    'FC10 failure axis2': torque_areaUnderSignal_FC10,
    'FC2 failure axis6': torque_areaUnderSignal_FC2,
    'FC1 failure axis2': torque_areaUnderSignal_FC1,
    'FC6 failure axis4': torque_areaUnderSignal_FC6,
    'FC7 failure axis2': torque_areaUnderSignal_FC7,
    'FC8 failure axis3': torque_areaUnderSignal_FC8,
    'FC11 failure axis2': torque_areaUnderSignal_FC11,
    'FC12 failure axis5': torque_areaUnderSignal_FC12
}

for i, (fc, d) in enumerate(data.items()):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    for index in axis_list:
        ax.plot(d[index], label=index)
        ax.set_title(f'Failure Case {fc}')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('torque_areaUnderSignal value')
        ax.set_ylim(0, .3)
        ax.legend()
    if isinstance(Failure_period[i][0], list):
        for x in Failure_period[i]:
            ax.axvspan(x[0], x[1], alpha=0.2, color='red')
    else:
        ax.axvspan(Failure_period[i][0], Failure_period[i][1], alpha=0.2, color='red')
    ax.scatter(x = d.index, y =np.full(shape=len(d.index), fill_value=0.25),s=0.1, color='r', label= 'Existed Data Point' )
    ax.legend(loc= "best",fontsize="6")


plt.tight_layout()


# In[67]:


axis_list = ['axis1','axis2','axis3','axis4','axis5','axis6']
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(18,10))

data = {
    'FC10 failure axis2': Stv_FC10,
    'FC2 failure axis6': Stv_FC2,
    'FC1 failure axis2': Stv_FC1,
    'FC6 failure axis4': Stv_FC6,
    'FC7 failure axis2': Stv_FC7,
    'FC8 failure axis3': Stv_FC8,
    'FC11 failure axis2': Stv_FC11,
    'FC12 failure axis5': Stv_FC12
}

for i, (fc, d) in enumerate(data.items()):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    for index in axis_list:
        ax.plot(d[index], label=index)
        ax.set_title(f'Failure Case {fc}')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Stv value')
        ax.set_ylim(0, .6)
        ax.legend()
        
    if isinstance(Failure_period[i][0], list):
        for x in Failure_period[i]:
            ax.axvspan(x[0], x[1], alpha=0.2, color='red')
    else:
        ax.axvspan(Failure_period[i][0], Failure_period[i][1], alpha=0.2, color='red')
    ax.scatter(x = d.index, y =np.full(shape=len(d.index), fill_value=0.4),s=0.1, color='r', label= 'Existed Data Point' )
    ax.legend(loc= "best",fontsize="6")

plt.tight_layout()


# In[68]:


axis_list = ['axis1','axis2','axis3','axis4','axis5','axis6']
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(18,10))

data = {
    'FC10 failure axis2': torque_trimmedMean_FC10,
    'FC2 failure axis6': torque_trimmedMean_FC2,
    'FC1 failure axis2': torque_trimmedMean_FC1,
    'FC6 failure axis4': torque_trimmedMean_FC6,
    'FC7 failure axis2': torque_trimmedMean_FC7,
    'FC8 failure axis3': torque_trimmedMean_FC8,
    'FC11 failure axis2': torque_trimmedMean_FC11,
    'FC12 failure axis5': torque_trimmedMean_FC12
}



for i, (fc, d) in enumerate(data.items()):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    for index in axis_list:
        

        
        ax.plot(d[index], label=index)
        ax.set_title(f'Failure Case {fc}')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('torque_trimmedMean value')
        ax.set_ylim(0, .2)
        ax.legend()
    if isinstance(Failure_period[i][0], list):
        for x in Failure_period[i]:
            ax.axvspan(x[0], x[1], alpha=0.2, color='red')
    else:
        ax.axvspan(Failure_period[i][0], Failure_period[i][1], alpha=0.2, color='red')
    
    ax.scatter(x = d.index, y =np.full(shape=len(d.index), fill_value=0.25),s=0.1, color='r', label= 'Existed Data Point' )
    ax.legend(loc= "best",fontsize="6")


plt.tight_layout()


# In[69]:


torque_trimmedMean_FC12 = torque_trimmedMean_FC12.dropna(axis=0,how='any')
torque_signalNoise_FC12 = torque_signalNoise_FC12.dropna(axis=0,how='any')
MAX_AMP_FC12 = MAX_AMP_FC12.dropna(axis=0,how='any')


# In[70]:


axis_list = ['axis1','axis2','axis3','axis4','axis5','axis6']
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(18,10))

data = {
    'FC10 failure axis2':MAX_AMP_FC10,
    'FC2 failure axis6': MAX_AMP_FC2,
    'FC1 failure axis2': MAX_AMP_FC1,
    'FC6 failure axis4': MAX_AMP_FC6,
    'FC7 failure axis2': MAX_AMP_FC7,
    'FC8 failure axis3': MAX_AMP_FC8,
    'FC11 failure axis2': MAX_AMP_FC11,
    'FC12 failure axis5': MAX_AMP_FC12
}

for i, (fc, d) in enumerate(data.items()):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    for index in axis_list:
        ax.plot(d[index], label=index)
        ax.set_title(f'Failure Case {fc}')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('MAX_AMP value')
#         ax.set_ylim(0, .2)
        ax.legend()
    if isinstance(Failure_period[i][0], list):
        for x in Failure_period[i]:
            ax.axvspan(x[0], x[1], alpha=0.2, color='red')
    else:
        ax.axvspan(Failure_period[i][0], Failure_period[i][1], alpha=0.2, color='red')
    
plt.legend()
plt.tight_layout()


# In[72]:


axis_list = ['axis1','axis2','axis3','axis4','axis5','axis6']
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(18,10))

data = {
    'FC10 failure axis2':torque_signalNoise_FC10,
    'FC2 failure axis6': torque_signalNoise_FC2,
    'FC1 failure axis2': torque_signalNoise_FC1,
    'FC6 failure axis4': torque_signalNoise_FC6,
    'FC7 failure axis2': torque_signalNoise_FC7,
    'FC8 failure axis3': torque_signalNoise_FC8,
    'FC11 failure axis2': torque_signalNoise_FC11,
    'FC12 failure axis5': torque_signalNoise_FC12
}

for i, (fc, d) in enumerate(data.items()):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    for index in axis_list:
        ax.plot(d[index], label=index)
        ax.set_title(f'Failure Case {fc}')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('torque_signalNoise value')
#         ax.set_ylim(0, .2)
        ax.legend()
    ax.axvspan(Failure_period[i][0], Failure_period[i][1], alpha=0.2, color='red')
    plt.legend()
plt.tight_layout()


# # Features for each axis

# In[73]:


feature_axis1_FC2 = pd.concat([RMS_Value_FC2['axis1'],Stv_FC2['axis1'],torque_areaUnderSignal_FC2['axis1']],axis=1)
feature_axis1_FC2.columns = ['RMS','Stv','areaUnderSignal']

feature_axis1_FC10 = pd.concat([RMS_Value_FC10['axis1'],Stv_FC10['axis1'],torque_areaUnderSignal_FC10['axis1']],axis=1)
feature_axis1_FC10.columns = ['RMS','Stv','areaUnderSignal']

feature_axis2_FC2 = pd.concat([RMS_Value_FC2['axis2'],Stv_FC2['axis2'],torque_areaUnderSignal_FC2['axis2']],axis=1)
feature_axis2_FC2.columns = ['RMS','Stv','areaUnderSignal']

feature_axis2_FC10 = pd.concat([RMS_Value_FC10['axis2'],Stv_FC10['axis2'],torque_areaUnderSignal_FC10['axis2']],axis=1)
feature_axis2_FC10.columns = ['RMS','Stv','areaUnderSignal']

feature_axis3_FC2 = pd.concat([RMS_Value_FC2['axis3'],Stv_FC2['axis3'],torque_areaUnderSignal_FC2['axis3']],axis=1)
feature_axis3_FC2.columns = ['RMS','Stv','areaUnderSignal']

feature_axis3_FC10 = pd.concat([RMS_Value_FC10['axis3'],Stv_FC10['axis3'],torque_areaUnderSignal_FC10['axis3']],axis=1)
feature_axis3_FC10.columns = ['RMS','Stv','areaUnderSignal']

feature_axis4_FC2 = pd.concat([RMS_Value_FC2['axis4'],Stv_FC2['axis4'],torque_areaUnderSignal_FC2['axis4']],axis=1)
feature_axis4_FC2.columns = ['RMS','Stv','areaUnderSignal']

feature_axis4_FC10 = pd.concat([RMS_Value_FC10['axis4'],Stv_FC10['axis4'],torque_areaUnderSignal_FC10['axis4']],axis=1)
feature_axis4_FC10.columns = ['RMS','Stv','areaUnderSignal']

feature_axis5_FC2 = pd.concat([RMS_Value_FC2['axis5'],Stv_FC2['axis5'],torque_areaUnderSignal_FC2['axis5']],axis=1)
feature_axis5_FC2.columns = ['RMS','Stv','areaUnderSignal']

feature_axis5_FC10 = pd.concat([RMS_Value_FC10['axis5'],Stv_FC10['axis5'],torque_areaUnderSignal_FC10['axis5']],axis=1)
feature_axis5_FC10.columns = ['RMS','Stv','areaUnderSignal']

feature_axis6_FC2 = pd.concat([RMS_Value_FC2['axis6'],Stv_FC2['axis6'],torque_areaUnderSignal_FC2['axis6']],axis=1)
feature_axis6_FC2.columns = ['RMS','Stv','areaUnderSignal']

feature_axis6_FC10 = pd.concat([RMS_Value_FC10['axis6'],Stv_FC10['axis6'],torque_areaUnderSignal_FC10['axis6']],axis=1)
feature_axis6_FC10.columns = ['RMS','Stv','areaUnderSignal']


# In[74]:




Failure_period = [line_FC10, line_FC2, line_FC1, line_FC6, line_FC7, line_FC8,line_FC11, line_FC12]


# In[75]:


data1 = feature_axis1_FC2
data2 = feature_axis2_FC2
data3 = feature_axis3_FC2
data4 = feature_axis4_FC2
data5 = feature_axis5_FC2
data6 = feature_axis6_FC2
frames_FC2 = [data1, data2, data3, data4, data5, data6]


# In[76]:


from mpl_toolkits.mplot3d import Axes3D
index1 = date(2015, 3, 20)
index2 = date(2015, 4, 3)
fig = plt.figure() 
ax = fig.add_subplot(projection='3d') 


ax.scatter(data1['RMS'], data1['Stv'], data1['areaUnderSignal'], marker = 'v', c='y')
ax.scatter(data2['RMS'], data2['Stv'], data2['areaUnderSignal'], marker = 's', c='k')
ax.scatter(data3['RMS'], data3['Stv'], data3['areaUnderSignal'], marker = 'p', c='c')
ax.scatter(data4['RMS'], data4['Stv'], data4['areaUnderSignal'], marker = '*', c='m')
ax.scatter(data5['RMS'], data5['Stv'], data5['areaUnderSignal'], marker = '^', c='g')
ax.scatter(data6['RMS'].loc[:index1], data6['Stv'].loc[:index1], data6['areaUnderSignal'].loc[:index1], c='b')
ax.scatter(data6['RMS'].loc[index2:], data6['Stv'].loc[index2:], data6['areaUnderSignal'].loc[index2:], c='b')
ax.scatter(data6['RMS'].loc[index1:index2], data6['Stv'].loc[index1:index2], data6['areaUnderSignal'].loc[index1:index2], c='r', marker='o', label='Anomaly data point')

ax.set_xlabel('RMS') 
ax.set_ylabel('Stv') 
ax.set_zlabel('areaUnderSignal') 
ax.legend()
ax.set_title('Failure Case 2')
plt.show()
# datadate(2019, 4, 18),date(2019, 6, 5)



# In[77]:


from mpl_toolkits.mplot3d import Axes3D
index1 = date(2015, 3, 20)
index2 = date(2015, 4, 3)
fig = plt.figure() 
ax = fig.add_subplot(projection='3d')


Index = pd.to_datetime(feature_axis1_FC2.index)
Markers = ['x','v','s','p','o','*']
axis_list = ['axis1','axis2','axis3','axis4','axis5','axis6']
i = 0
for data in frames_FC2:
    sca = ax.scatter(data['RMS'], data['Stv'], data['areaUnderSignal'], c=Index, marker = Markers[i],cmap='jet_r', label = axis_list[i])
    i = i + 1

cbar = plt.colorbar(sca)

cbar.ax.set_yticklabels(pd.to_datetime(cbar.get_ticks()).strftime(date_format='%b %Y'))
ax.set_xlabel('RMS') 
ax.set_ylabel('Stv') 
ax.set_zlabel('areaUnderSignal') 
ax.legend()
ax.set_title('Failure Case 2')
plt.show()
# datadate(2019, 4, 18),date(2019, 6, 5)




# In[78]:


data1 = feature_axis1_FC10
data2 = feature_axis2_FC10
data3 = feature_axis3_FC10
data4 = feature_axis4_FC10
data5 = feature_axis5_FC10
data6 = feature_axis6_FC10
frames_FC10 = [data1, data2, data3, data4, data5, data6]

from mpl_toolkits.mplot3d import Axes3D
index1 = date(2019, 4, 18)
index2 = date(2019, 6, 5)
fig = plt.figure() 
ax = fig.add_subplot(projection='3d') 


ax.scatter(data1['RMS'], data1['Stv'], data1['areaUnderSignal'], c='g',marker = 'o')

ax.scatter(data3['RMS'], data3['Stv'], data3['areaUnderSignal'], c='y',marker = 'v')
ax.scatter(data4['RMS'], data4['Stv'], data4['areaUnderSignal'], c='k',marker = 's')
ax.scatter(data5['RMS'], data5['Stv'], data5['areaUnderSignal'], c='m', marker = 'p')
ax.scatter(data6['RMS'], data6['Stv'], data6['areaUnderSignal'], c='c',marker = '*')
# ax.scatter(data2['RMS'], data2['Stv'], data2['areaUnderSignal'], c='b')
ax.scatter(data2['RMS'].loc[:index1], data2['Stv'].loc[:index1], data2['areaUnderSignal'].loc[:index1], c='b')
ax.scatter(data2['RMS'].loc[index2:], data2['Stv'].loc[index2:], data2['areaUnderSignal'].loc[index2:], c='b')

ax.scatter(data2['RMS'].loc[index1:index2], data2['Stv'].loc[index1:index2], data2['areaUnderSignal'].loc[index1:index2], c='r', marker='o', label='Anomaly data point')
ax.set_xlabel('RMS')
ax.set_ylabel('Stv')
ax.set_zlabel('areaUnderSignal')
ax.legend()
ax.set_title('Failure Case 10')
plt.show()
# datadate(2019, 4, 18),date(2019, 6, 5)



# In[79]:



fig = plt.figure() 
ax = fig.add_subplot(projection='3d')


Index = pd.to_datetime(feature_axis1_FC10.index)
Markers = ['x','v','s','p','o','*']
axis_list = ['axis1','axis2','axis3','axis4','axis5','axis6']
i = 0
for data in frames_FC10:
    sca = ax.scatter(data['RMS'], data['Stv'], data['areaUnderSignal'], c=Index, marker = Markers[i],cmap='jet_r', label = axis_list[i])
    i = i + 1

cbar = plt.colorbar(sca)

cbar.ax.set_yticklabels(pd.to_datetime(cbar.get_ticks()).strftime(date_format='%b %Y'))
ax.set_xlabel('RMS') 
ax.set_ylabel('Stv') 
ax.set_zlabel('areaUnderSignal') 
ax.legend()
ax.set_title('Failure Case 10')
plt.show()
# datadate(2019, 4, 18),date(2019, 6, 5)


# # Aggregating Features

# 1. Distance between the centriod of the reference cluster and the updated data points

# In[80]:


# Global
def centriod_distance(ref_cluster,new):
    x = ref_cluster['RMS']
    y = ref_cluster['Stv']
    z = ref_cluster['areaUnderSignal']
    middle = np.average(x), np.average(y), np.average(z)
    D = np.linalg.norm(new - middle)
    return D
# Get the reference cluster
def plot_distance_local(frames,k,failure_Case,ax,Failure_period):
    
    
    ref_cluster_FCX = {
    'axis1': frames[0][0:k],
    'axis2': frames[1][0:k],
    'axis3': frames[2][0:k],
    'axis4': frames[3][0:k],
    'axis5': frames[4][0:k],
    'axis6': frames[5][0:k],

    }

    
    data = {
        'axis1': frames[0][k+1:],
        'axis2': frames[1][k+1:],
        'axis3': frames[2][k+1:],
        'axis4': frames[3][k+1:],
        'axis5': frames[4][k+1:],
        'axis6': frames[5][k+1:],

    }
    
    Dist = [[],[],[],[],[],[]]
    Distance_FCX = {}
    for i, (fc, d) in enumerate(data.items()):
        x = pd.to_datetime(d.index)
        for index, row in d.iterrows():
            Dist[i].append(centriod_distance(ref_cluster_FCX[fc],row))
        Distance_FCX[fc] = Dist[i]

        ax.plot(x, Distance_FCX[fc],label = fc)
    if isinstance(Failure_period[0], list):
        for x in Failure_period:
            ax.axvspan(x[0], x[1], alpha=0.2,  color='yellow',label='failure period')
    else:
        ax.axvspan(Failure_period[0], Failure_period[1], alpha=0.2, color='yellow')
#     ax.set_ylim(0, 0.5)
    ax.set_title(failure_Case)
    ax.legend()
    return Distance_FCX
    
    
def plot_distance_global(frames,k,failure_Case,ax,Failure_period):
        
    ref_cluster_FCX = pd.concat([frames[0][0:10],
                                 frames[1][0:10],
                                 frames[2][0:10],
                                 frames[3][0:10],
                                 frames[4][0:10],
                                 frames[5][0:10]],axis = 0)

    
    data = {
        'axis1': frames[0][k+1:],
        'axis2': frames[1][k+1:],
        'axis3': frames[2][k+1:],
        'axis4': frames[3][k+1:],
        'axis5': frames[4][k+1:],
        'axis6': frames[5][k+1:],

    }
    
    Dist = [[],[],[],[],[],[]]
    Distance_FCX = {}
    for i, (fc, d) in enumerate(data.items()):
        x = pd.to_datetime(d.index)
        for index, row in d.iterrows():
            Dist[i].append(centriod_distance(ref_cluster_FCX,row))
        Distance_FCX[fc] = Dist[i]

        ax.plot(x, Distance_FCX[fc],label = fc)
    if isinstance(Failure_period[0], list):
        for x in Failure_period:
            ax.axvspan(x[0], x[1], alpha=0.2,  color='yellow',label='failure period')
    else:
        ax.axvspan(Failure_period[0], Failure_period[1], alpha=0.2, color='yellow',label='failure period')
#     ax.set_ylim(0, 0.5)
    ax.set_title(failure_Case)
    ax.legend()
    return Distance_FCX


# In[117]:


data1 = feature_axis1_FC2
data2 = feature_axis2_FC2
data3 = feature_axis3_FC2
data4 = feature_axis4_FC2
data5 = feature_axis5_FC2
data6 = feature_axis6_FC2
frames_FC2 = [data1, data2, data3, data4, data5, data6]

line_FC2 = [date(2015, 3, 20),date(2015, 4, 3)]
fig, (ax1, ax2) = plt.subplots(2,figsize=(18,10))
Distance_FC2_Global = plot_distance_global(frames_FC2,10,'FC2 Centroid-based global',ax1,line_FC2)

Distance_FC2_Local = plot_distance_local(frames_FC2,10, 'FC2 Centroid-based local',ax2,line_FC2)


# In[121]:



import pymannkendall as mk
window_size = 20
Trend = []
for i in range(0, len(Distance_FC2_Global['axis3']), window_size):
    end_index = i + window_size
    data = Distance_FC2_Global['axis4'][i:end_index]
    trend, *_ = mk.original_test(data)
    Trend.append(trend)
Trend


# In[84]:


data1 = feature_axis1_FC10
data2 = feature_axis2_FC10
data3 = feature_axis3_FC10
data4 = feature_axis4_FC10
data5 = feature_axis5_FC10
data6 = feature_axis6_FC10
frames_FC10 = [data1, data2, data3, data4, data5, data6]

line_FC10 = [date(2019, 4, 18),date(2019, 6, 5)]
fig, (ax1, ax2) = plt.subplots(2,figsize=(18,10))
Distance_FC10_Global = plot_distance_global(frames_FC10,10,'FC10 Centroid-based global',ax1,line_FC10)

Distance_FC10_Local = plot_distance_local(frames_FC10,10, 'FC10 Centroid-based local',ax2,line_FC10)


# # Moving Average Window(Trend Detection)

# In[86]:


data1 = feature_axis1_FC2
data2 = feature_axis2_FC2
data3 = feature_axis3_FC2
data4 = feature_axis4_FC2
data5 = feature_axis5_FC2
data6 = feature_axis6_FC2
frames_FC2 = [data1, data2, data3, data4, data5, data6]
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(18,10))
k = 10
Mean = []
std = []
columns=['axis1','axis2','axis3','axis4','axis5','axis6']
Distance_Global_FC2 =pd.DataFrame(columns=columns)
for i, (fc, d) in enumerate(Distance_FC2_Global.items()):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    x = pd.to_datetime(data1[k+1:].index)
    Distance_Global_FC2[fc] = d
    Average = Distance_Global_FC2[fc].rolling(7).mean()
    Moved_average = np.mean(d) + np.std(d)*0.6
    ax.axhline(y = Moved_average, color = 'r', linestyle = 'dashed',label= 'Moved average (Multiplier = 0.6)')
    ax.plot(x, Average,label = fc)
    if fc == 'axis6':
        ax.set_title(f'Failure Case 2 {fc}',color = 'r')
    else:
        ax.set_title(f'Failure Case 2 {fc}')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Centroid-based Value')
    ax.set_ylim(0, .5)
    ax.legend(loc= "best",fontsize="6")
    
plt.tight_layout()


# In[85]:


fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(18,10))
k = 10
Mean = []
std = []
columns=['axis1','axis2','axis3','axis4','axis5','axis6']
Distance_Global_FC10 =pd.DataFrame(columns=columns)
for i, (fc, d) in enumerate(Distance_FC10_Global.items()):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    x = pd.to_datetime(data1[k+1:].index)
    Distance_Global_FC10[fc] = d
    Average = Distance_Global_FC10[fc].rolling(7).mean()
    Moved_average = np.mean(d) + np.std(d)*0.6
    ax.axhline(y = Moved_average, color = 'r', linestyle = '-',label= 'Moved average (Multiplier = 0.5)')
    ax.plot(x, Average,label = fc)
    if fc == 'axis2':
        ax.set_title(f'Failure Case 10 {fc}',color = 'r')
    else:
        ax.set_title(f'Failure Case 10 {fc}')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Centroid-based Value')
    ax.set_ylim(0, .5)
    ax.legend(loc= "best",fontsize="6")
    
plt.tight_layout()


# In[ ]:


window_size = 14
Trend = []
for i in range(0, len(Distance_FC10_Global['axis1']), window_size):
    end_index = i + window_size
    data = Distance_FC10_Global['axis2'][i:end_index]
    trend, *_ = mk.original_test(data)
    Trend.append(trend)

    
Trend


# # Distance-based local oulier

# In[122]:


# Failure Case 2
from scipy.spatial.distance import pdist, squareform
def outer_distance(ref_cluster,new):
    Dist = []
    for index, row in ref_cluster.iterrows():
        Dist.append(np.linalg.norm(new - row))
    d_outer = sum(Dist)/len(ref_cluster)
    return d_outer

def Inner_distance(ref_cluster):
    distances = pdist(ref_cluster)
    k = len(ref_cluster)
    d_inner = sum(distances)/(k*(k-1))
    return d_inner

def Distance_based_lof_global(frames,k, title, ax, Failure_period):

    ref_cluster_FCX = pd.concat([frames[0][0:k],
                                     frames[1][0:k],
                                     frames[2][0:k],
                                     frames[3][0:k],
                                     frames[4][0:k],
                                     frames[5][0:k]],axis = 0)

    Inner_dist_FCX = Inner_distance(ref_cluster_FCX)
    #The rest data points
    data = {
        'axis1': frames[0][k+1:],
        'axis2': frames[1][k+1:],
        'axis3': frames[2][k+1:],
        'axis4': frames[3][k+1:],
        'axis5': frames[4][k+1:],
        'axis6': frames[5][k+1:],

    }

    Dist = [[],[],[],[],[],[]]
    Distance_FCX = {}

    for i, (fc, d) in enumerate(data.items()):
        x = pd.to_datetime(d.index)
        for index, row in d.iterrows():
            Dist[i].append(outer_distance(ref_cluster_FCX,row)/Inner_dist_FCX)
        Distance_FCX[fc] = Dist[i]
        ax.plot(x, Distance_FCX[fc],label = fc)
    if isinstance(Failure_period[0], list):
        for x in Failure_period:
            ax.axvspan(x[0], x[1], alpha=0.2, color='yellow',label='failure period')
    else:
        ax.axvspan(Failure_period[0], Failure_period[1], alpha=0.2, color='yellow',label='failure period')
    ax.set_title(title)
    ax.legend()
    return Distance_FCX
        
def Distance_based_lof_local(frames,k, title, ax, Failure_period):

    ref_cluster_FCX = {
    'axis1': frames[0][0:k],
    'axis2': frames[1][0:k],
    'axis3': frames[2][0:k],
    'axis4': frames[3][0:k],
    'axis5': frames[4][0:k],
    'axis6': frames[5][0:k],
    }
    
    Inner_dist_FCX = {}
    for i, (fc, d) in enumerate(ref_cluster_FCX.items()):
        Inner_dist_FCX[fc] = Inner_distance(d)

    #The rest data points
    data = {
            'axis1': frames[0][k+1:],
            'axis2': frames[1][k+1:],
            'axis3': frames[2][k+1:],
            'axis4': frames[3][k+1:],
            'axis5': frames[4][k+1:],
            'axis6': frames[5][k+1:],
    }


    Dist = [[],[],[],[],[],[]]
    Distance_FCX = {}

    for i, (fc, d) in enumerate(data.items()):
        x = pd.to_datetime(d.index)
        for index, row in d.iterrows():
            Dist[i].append(outer_distance(ref_cluster_FCX[fc],row)/Inner_dist_FCX[fc])
        Distance_FCX[fc] = Dist[i]
        ax.plot(x, Distance_FCX[fc],label = fc)
    if isinstance(Failure_period[0], list):
        for x in Failure_period:
            ax.axvspan(x[0], x[1], alpha=0.2, color='yellow',label='failure period')
    else:
        ax.axvspan(Failure_period[0], Failure_period[1], alpha=0.2, color='yellow',label='failure period')
    ax.set_title(title)
    ax.legend()
    return Distance_FCX
        


# In[123]:


data1 = feature_axis1_FC2
data2 = feature_axis2_FC2
data3 = feature_axis3_FC2
data4 = feature_axis4_FC2
data5 = feature_axis5_FC2
data6 = feature_axis6_FC2
frames_FC2 = [data1, data2, data3, data4, data5, data6]
fig, (ax1, ax2) = plt.subplots(2,figsize=(18,10))
Distance_based_lof_global_FC2 = Distance_based_lof_global(frames_FC2,10,'global distance_based LOF FC2',ax1,line_FC2)
Distance_based_lof_local_FC2 = Distance_based_lof_local(frames_FC2,10,'local distance_based LOF FC2',ax2,line_FC2)
# plot_distance_local(frames_FC2,10, 'FC2 Centroid-based local',ax2)


# In[ ]:


fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(18,10))
k = 10
Mean = []
std = []
columns=['axis1','axis2','axis3','axis4','axis5','axis6']
Distance_Global_FC10 =pd.DataFrame(columns=columns)
for i, (fc, d) in enumerate(Distance_FC10_Global.items()):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    x = pd.to_datetime(data1[k+1:].index)
    Distance_Global_FC10[fc] = d
    Average = Distance_Global_FC10[fc].rolling(7).mean()
    Moved_average = np.mean(d) + np.std(d)*0.6
    ax.axhline(y = Moved_average, color = 'r', linestyle = '-',label= 'Moved average (Multiplier = 0.5)')
    ax.plot(x, Average,label = fc)
    if fc == 'axis2':
        ax.set_title(f'Failure Case 10 {fc}',color = 'r')
    else:
        ax.set_title(f'Failure Case 10 {fc}')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Centroid-based Value')
    ax.set_ylim(0, .5)
    ax.legend(loc= "best",fontsize="6")
    
plt.tight_layout()


# In[147]:


data1 = feature_axis1_FC10
data2 = feature_axis2_FC10
data3 = feature_axis3_FC10
data4 = feature_axis4_FC10
data5 = feature_axis5_FC10
data6 = feature_axis6_FC10
frames_FC10 = [data1, data2, data3, data4, data5, data6]
fig, (ax1, ax2) = plt.subplots(2,figsize=(18,10))
Distance_based_lof_global(frames_FC10,10,'global distance_based LOF FC10',ax1,line_FC10)
Distance_based_lof_local(frames_FC10,10,'local distance_based LOF FC10',ax2,line_FC10)


# # Local Outlier Factor VS Isolation Forest

# In[170]:


from sklearn.neighbors import LocalOutlierFactor


def classic_LOF(frames, k, title, failure_period):
    k = 10
    reference = pd.concat([frames[0][0:k],
                                         frames[1][0:k],
                                         frames[2][0:k],
                                         frames[3][0:k],
                                         frames[4][0:k],
                                         frames[5][0:k]],axis = 0)

    fig, ax = plt.subplots(1,figsize=(18,10))
    # Train the local outlier factor (LOF) model for novelty detection
    lof_novelty = LocalOutlierFactor(n_neighbors=5, novelty=True).fit(reference)
    # Predict novelties
    data = {
            'axis1': frames[0][k+1:],
            'axis2': frames[1][k+1:],
            'axis3': frames[2][k+1:],
            'axis4': frames[3][k+1:],
            'axis5': frames[4][k+1:],
            'axis6': frames[5][k+1:],

        }

    Dist = [[],[],[],[],[],[]]
    Distance_FCX = {}

    for i, (fc, d) in enumerate(data.items()):
        x = pd.to_datetime(data1[k+1:].index)
 
        for index, row in d.iterrows():
            row = row.to_numpy()
            row = row.reshape(1,3)
            prediction_novelty = lof_novelty.predict(row)
            score = lof_novelty.score_samples(row)
            Dist[i].append(abs(score))
        Distance_FCX[fc] = Dist[i]
        ax.plot(x, Distance_FCX[fc],label = fc)
    if isinstance(failure_period[0], list):
        for x in failure_period:
            ax.axvspan(x[0], x[1], alpha=0.2, color='yellow',label='failure period')
    else:
        ax.axvspan(failure_period[0], failure_period[1], alpha=0.2, color='yellow',label='failure period')

    ax.set_title(title)
    ax.legend()
    return Distance_FCX
    


# In[171]:


data1 = feature_axis1_FC2
data2 = feature_axis2_FC2
data3 = feature_axis3_FC2
data4 = feature_axis4_FC2
data5 = feature_axis5_FC2
data6 = feature_axis6_FC2
frames_FC2 = [data1, data2, data3, data4, data5, data6]
classic_LOF_global_FC2 = classic_LOF(frames_FC2,10,'global novelty detection LOF FC2',line_FC2)


# In[172]:


data1 = feature_axis1_FC10
data2 = feature_axis2_FC10
data3 = feature_axis3_FC10
data4 = feature_axis4_FC10
data5 = feature_axis5_FC10
data6 = feature_axis6_FC10
frames_FC10 = [data1, data2, data3, data4, data5, data6]
classic_LOF_global_FC10 = classic_LOF(frames_FC10,10,'global novelty detection LOF FC10',line_FC10)


# In[177]:


from sklearn.svm import OneClassSVM


def OneclassSVM(frames, k, title, failure_period):
    gamma = 2.0
    reference = pd.concat([frames[0][0:k],
                                         frames[1][0:k],
                                         frames[2][0:k],
                                         frames[3][0:k],
                                         frames[4][0:k],
                                         frames[5][0:k]],axis = 0)

    fig, ax = plt.subplots(1,figsize=(18,10))
    # Train the local outlier factor (LOF) model for novelty detection
    OneclassSVM_novelty = OneClassSVM(gamma=gamma, kernel="rbf", nu=0.05).fit(reference)
    # Predict novelties
    data = {
            'axis1': frames[0][k+1:],
            'axis2': frames[1][k+1:],
            'axis3': frames[2][k+1:],
            'axis4': frames[3][k+1:],
            'axis5': frames[4][k+1:],
            'axis6': frames[5][k+1:],

        }

    Dist = [[],[],[],[],[],[]]
    Distance_FCX = {}

    for i, (fc, d) in enumerate(data.items()):
        x = pd.to_datetime(data1[k+1:].index)

        for index, row in d.iterrows():
            row = row.to_numpy()
            row = row.reshape(1,3)
            prediction_novelty = OneclassSVM_novelty.predict(row)
            score = OneclassSVM_novelty.score_samples(row)
            Dist[i].append(1/score)
        Distance_FCX[fc] = Dist[i]
        ax.plot(x, Distance_FCX[fc],label = fc)
    if isinstance(failure_period[0], list):
        for x in failure_period:
            ax.axvspan(x[0], x[1], alpha=0.2, color='yellow',label='failure period')
    else:
        ax.axvspan(failure_period[0], failure_period[1], alpha=0.2, color='yellow',label='failure period')

    ax.set_title(title)
    ax.legend()
    return Distance_FCX


# In[178]:


data1 = feature_axis1_FC2
data2 = feature_axis2_FC2
data3 = feature_axis3_FC2
data4 = feature_axis4_FC2
data5 = feature_axis5_FC2
data6 = feature_axis6_FC2
frames_FC2 = [data1, data2, data3, data4, data5, data6]
OneclassSVM_global_FC2 = OneclassSVM(frames_FC2,10,'global novelty detection LOF FC2',line_FC2)


# In[179]:


data1 = feature_axis1_FC10
data2 = feature_axis2_FC10
data3 = feature_axis3_FC10
data4 = feature_axis4_FC10
data5 = feature_axis5_FC10
data6 = feature_axis6_FC10
frames_FC10 = [data1, data2, data3, data4, data5, data6]
OneclassSVM_global_FC2 = OneclassSVM(frames_FC10,10,'global novelty detection OneclassSVM FC10',line_FC10)


# # Autoencoder

# In[111]:


import tensorflow


# In[112]:


from pyod.models.auto_encoder import AutoEncoder


def autoencoder(frames,title,k,Failure_period):
    X_train = pd.concat([frames[0][0:10],
                                     frames[1][0:k],
                                     frames[2][0:k],
                                     frames[3][0:k],
                                     frames[4][0:k],
                                     frames[5][0:k]],axis = 0)
    X_test = {
        'axis1': frames[0][k+1:],
        'axis2': frames[1][k+1:],
        'axis3': frames[2][k+1:],
        'axis4': frames[3][k+1:],
        'axis5': frames[4][k+1:],
        'axis6': frames[5][k+1:],

    }

    fig = plt.figure(figsize=(18, 9))
    ax = fig.add_subplot(1, 1, 1)
    # train AutoEncoder detector
    clf_name = 'AutoEncoder'
    clf = AutoEncoder(epochs=100, hidden_neurons =[4, 2, 2, 4])
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)

    y_train_scores = clf.decision_scores_  # Reconstruction error

    # get the prediction on the test data

    Error = [[],[],[],[],[],[]]
    Error_FC10 = {}

    for i, (fc, d) in enumerate(X_test.items()):
        x = pd.to_datetime(d.index)
        for index, row in d.iterrows():
            row = row.to_numpy()
            row = row.reshape(1,3)
            Error[i].append(clf.decision_function(row))
        Error_FC10[fc] = Error[i]
        ax.plot(x, Error_FC10[fc],label = fc)
    if isinstance(Failure_period[0], list):
        for x in Failure_period:
            ax.axvspan(x[0], x[1], alpha=0.2, color='yellow',label='failure period')
    else:
        ax.axvspan(Failure_period[0], Failure_period[1], alpha=0.2, color='yellow',label='failure period')
#         ax.set_ylim(0, 20)
    ax.set_title(title)
    ax.legend()


# In[113]:


data1 = feature_axis1_FC10
data2 = feature_axis2_FC10
data3 = feature_axis3_FC10
data4 = feature_axis4_FC10
data5 = feature_axis5_FC10
data6 = feature_axis6_FC10
frames = [data1, data2, data3, data4, data5, data6]
autoencoder(frames,'Autoencoder FC10',20,line_FC10)


# In[114]:


data1 = feature_axis1_FC2
data2 = feature_axis2_FC2
data3 = feature_axis3_FC2
data4 = feature_axis4_FC2
data5 = feature_axis5_FC2
data6 = feature_axis6_FC2
frames = [data1, data2, data3, data4, data5, data6]
autoencoder(frames,'Autoencoder FC2',20,line_FC2)


# In[93]:


feature_axis1_FC1 = pd.concat([RMS_Value_FC1['axis1'],Stv_FC1['axis1'],torque_areaUnderSignal_FC1['axis1']],axis=1)
feature_axis1_FC1.columns = ['RMS','Stv','areaUnderSignal']

feature_axis1_FC6 = pd.concat([RMS_Value_FC6['axis1'],Stv_FC6['axis1'],torque_areaUnderSignal_FC6['axis1']],axis=1)
feature_axis1_FC6.columns = ['RMS','Stv','areaUnderSignal']

feature_axis2_FC1 = pd.concat([RMS_Value_FC1['axis2'],Stv_FC1['axis2'],torque_areaUnderSignal_FC1['axis2']],axis=1)
feature_axis2_FC1.columns = ['RMS','Stv','areaUnderSignal']

feature_axis2_FC6 = pd.concat([RMS_Value_FC6['axis2'],Stv_FC6['axis2'],torque_areaUnderSignal_FC6['axis2']],axis=1)
feature_axis2_FC6.columns = ['RMS','Stv','areaUnderSignal']

feature_axis3_FC1 = pd.concat([RMS_Value_FC1['axis3'],Stv_FC1['axis3'],torque_areaUnderSignal_FC1['axis3']],axis=1)
feature_axis3_FC1.columns = ['RMS','Stv','areaUnderSignal']

feature_axis3_FC6 = pd.concat([RMS_Value_FC6['axis3'],Stv_FC6['axis3'],torque_areaUnderSignal_FC6['axis3']],axis=1)
feature_axis3_FC6.columns = ['RMS','Stv','areaUnderSignal']

feature_axis4_FC1 = pd.concat([RMS_Value_FC1['axis4'],Stv_FC1['axis4'],torque_areaUnderSignal_FC1['axis4']],axis=1)
feature_axis4_FC1.columns = ['RMS','Stv','areaUnderSignal']

feature_axis4_FC6 = pd.concat([RMS_Value_FC6['axis4'],Stv_FC6['axis4'],torque_areaUnderSignal_FC6['axis4']],axis=1)
feature_axis4_FC6.columns = ['RMS','Stv','areaUnderSignal']

feature_axis5_FC1 = pd.concat([RMS_Value_FC1['axis5'],Stv_FC1['axis5'],torque_areaUnderSignal_FC1['axis5']],axis=1)
feature_axis5_FC1.columns = ['RMS','Stv','areaUnderSignal']

feature_axis5_FC6 = pd.concat([RMS_Value_FC6['axis6'],Stv_FC6['axis5'],torque_areaUnderSignal_FC6['axis5']],axis=1)
feature_axis5_FC6.columns = ['RMS','Stv','areaUnderSignal']

feature_axis6_FC1 = pd.concat([RMS_Value_FC1['axis6'],Stv_FC1['axis6'],torque_areaUnderSignal_FC1['axis6']],axis=1)
feature_axis6_FC1.columns = ['RMS','Stv','areaUnderSignal']

feature_axis6_FC6 = pd.concat([RMS_Value_FC6['axis6'],Stv_FC6['axis6'],torque_areaUnderSignal_FC6['axis6']],axis=1)
feature_axis6_FC6.columns = ['RMS','Stv','areaUnderSignal']


# # Centroid-based

# In[94]:


line_FC1 = [date(2016, 5, 6),date(2016, 5, 15)]
line_FC6 = [date(2016, 10, 1),date(2017, 2, 4)]


# In[95]:



data1 = feature_axis1_FC1
data2 = feature_axis2_FC1
data3 = feature_axis3_FC1
data4 = feature_axis4_FC1
data5 = feature_axis5_FC1
data6 = feature_axis6_FC1
frames_FC1 = [data1, data2, data3, data4, data5, data6]

fig, (ax1, ax2) = plt.subplots(2,figsize=(18,10))
Distance_FC1_Global = plot_distance_global(frames_FC1,10,'FC1 Centroid-based global',ax1,line_FC1)

Distance_FC1_Local = plot_distance_local(frames_FC1,10, 'FC1 Centroid-based local',ax2,line_FC1)


# In[96]:


fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(18,10))
k = 10
Mean = []
std = []
columns=['axis1','axis2','axis3','axis4','axis5','axis6']
Distance_Global_FC1 =pd.DataFrame(columns=columns)
for i, (fc, d) in enumerate(Distance_FC1_Global.items()):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    x = pd.to_datetime(data1[k+1:].index)
    Distance_Global_FC1[fc] = d
    Average = Distance_Global_FC1[fc].rolling(7).mean()
    Moved_average = np.mean(d) + np.std(d)*0.6
    ax.axhline(y = Moved_average, color = 'r', linestyle = 'dashed',label= 'Moved average (Multiplier = 0.6)')
    ax.plot(x, Average,label = fc)
    if fc == 'axis2':
        ax.set_title(f'Failure Case 1 {fc}',color = 'r')
    else:
        ax.set_title(f'Failure Case 1 {fc}')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Centroid-based Value')
    ax.set_ylim(0, .5)
    ax.legend(loc= "best",fontsize="6")
    
plt.tight_layout()


# # distance_based LOF

# In[99]:


data1 = feature_axis1_FC1
data2 = feature_axis2_FC1
data3 = feature_axis3_FC1
data4 = feature_axis4_FC1
data5 = feature_axis5_FC1
data6 = feature_axis6_FC1
frames_FC1 = [data1, data2, data3, data4, data5, data6]
fig, (ax1, ax2) = plt.subplots(2,figsize=(18,10))
Distance_based_lof_global(frames_FC1,10,'global distance_based LOF FC1',ax1,line_FC1)
Distance_based_lof_local(frames_FC1,10,'local distance_based LOF FC1',ax2,line_FC1)
# plot_distance_local(frames_FC2,10, 'FC2 Centroid-based local',ax2)


# In[105]:



from mpl_toolkits.mplot3d import Axes3D
index1 = date(2016, 5, 5)
index2 = date(2016, 5, 15)
fig = plt.figure() 
ax = fig.add_subplot(projection='3d') 

data1 = feature_axis1_FC1
data2 = feature_axis2_FC1
data3 = feature_axis3_FC1
data4 = feature_axis4_FC1
data5 = feature_axis5_FC1
data6 = feature_axis6_FC1

ax.scatter(data1['RMS'], data1['Stv'], data1['areaUnderSignal'], c='g',marker = 'o')

ax.scatter(data3['RMS'], data3['Stv'], data3['areaUnderSignal'], c='y',marker = 'v')
ax.scatter(data4['RMS'], data4['Stv'], data4['areaUnderSignal'], c='k',marker = 's')
ax.scatter(data5['RMS'], data5['Stv'], data5['areaUnderSignal'], c='m', marker = 'p')
ax.scatter(data6['RMS'], data6['Stv'], data6['areaUnderSignal'], c='c',marker = '*')
# ax.scatter(data2['RMS'], data2['Stv'], data2['areaUnderSignal'], c='b')
ax.scatter(data2['RMS'].loc[:index1], data2['Stv'].loc[:index1], data2['areaUnderSignal'].loc[:index1], c='b')
ax.scatter(data2['RMS'].loc[index2:], data2['Stv'].loc[index2:], data2['areaUnderSignal'].loc[index2:], c='b')

ax.scatter(data2['RMS'].loc[index1:index2], data2['Stv'].loc[index1:index2], data2['areaUnderSignal'].loc[index1:index2], c='r', marker='o', label='Anomaly data point')
ax.set_xlabel('RMS')
ax.set_ylabel('Stv')
ax.set_zlabel('areaUnderSignal')
ax.legend()
ax.set_title('Failure Case 1')
plt.show()


# In[101]:


fig = plt.figure() 
ax = fig.add_subplot(projection='3d')


Index = pd.to_datetime(feature_axis1_FC1.index)
Markers = ['x','v','s','p','o','*']
axis_list = ['axis1','axis2','axis3','axis4','axis5','axis6']
i = 0
for data in frames_FC1:
    sca = ax.scatter(data['RMS'], data['Stv'], data['areaUnderSignal'], c=Index, marker = Markers[i],cmap='jet_r', label = axis_list[i])
    i = i + 1

cbar = plt.colorbar(sca)

cbar.ax.set_yticklabels(pd.to_datetime(cbar.get_ticks()).strftime(date_format='%b %Y'))
ax.set_xlabel('RMS') 
ax.set_ylabel('Stv') 
ax.set_zlabel('areaUnderSignal') 
ax.legend()
ax.set_title('Failure Case 1')
plt.show()


# In[102]:


data1 = feature_axis1_FC6
data2 = feature_axis2_FC6
data3 = feature_axis3_FC6
data4 = feature_axis4_FC6
data5 = feature_axis5_FC6
data6 = feature_axis6_FC6
frames_FC6 = [data1, data2, data3, data4, data5, data6]

fig, (ax1, ax2) = plt.subplots(2,figsize=(18,10))
Distance_FC6_Global = plot_distance_global(frames_FC6,10,'FC6 Centroid-based global',ax1,line_FC6)

Distance_FC6_Local = plot_distance_local(frames_FC6,10, 'FC6 Centroid-based local',ax2,line_FC6)


# In[103]:


fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(18,10))
k = 10
Mean = []
std = []
columns=['axis1','axis2','axis3','axis4','axis5','axis6']
Distance_Global_FC6 =pd.DataFrame(columns=columns)
for i, (fc, d) in enumerate(Distance_FC6_Global.items()):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    x = pd.to_datetime(data1[k+1:].index)
    Distance_Global_FC6[fc] = d
    Average = Distance_Global_FC6[fc].rolling(7).mean()
    Moved_average = np.mean(d) + np.std(d)*0.6
    ax.axhline(y = Moved_average, color = 'r', linestyle = 'dashed',label= 'Moved average (Multiplier = 0.6)')
    ax.plot(x, Average,label = fc)
    if fc == 'axis4':
        ax.set_title(f'Failure Case 6 {fc}',color = 'r')
    else:
        ax.set_title(f'Failure Case 6 {fc}')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Centroid-based Value')
    ax.set_ylim(0, .5)
    ax.legend(loc= "best",fontsize="6")
    
plt.tight_layout()


# In[106]:


data1 = feature_axis1_FC6
data2 = feature_axis2_FC6
data3 = feature_axis3_FC6
data4 = feature_axis4_FC6
data5 = feature_axis5_FC6
data6 = feature_axis6_FC6
frames_FC6 = [data1, data2, data3, data4, data5, data6]
fig, (ax1, ax2) = plt.subplots(2,figsize=(18,10))
Distance_based_lof_global(frames_FC6,10,'global distance_based LOF FC6',ax1,line_FC6)
Distance_based_lof_local(frames_FC6,10,'local distance_based LOF FC6',ax2,line_FC6)


# In[107]:



from mpl_toolkits.mplot3d import Axes3D
index1 = date(2016, 9, 9)
index2 = date(2017, 3, 3)
fig = plt.figure() 
ax = fig.add_subplot(projection='3d') 

ax.scatter(data1['RMS'], data1['Stv'], data1['areaUnderSignal'], c='g',marker = 'o')
ax.scatter(data2['RMS'], data2['Stv'], data2['areaUnderSignal'], c='k',marker = 's')
ax.scatter(data3['RMS'], data3['Stv'], data3['areaUnderSignal'], c='y',marker = 'v')

ax.scatter(data5['RMS'], data5['Stv'], data5['areaUnderSignal'], c='m', marker = 'p')
ax.scatter(data6['RMS'], data6['Stv'], data6['areaUnderSignal'], c='c',marker = '*')
# ax.scatter(data2['RMS'], data2['Stv'], data2['areaUnderSignal'], c='b')
ax.scatter(data4['RMS'].loc[:index1], data4['Stv'].loc[:index1], data4['areaUnderSignal'].loc[:index1], c='b')
ax.scatter(data4['RMS'].loc[index2:], data4['Stv'].loc[index2:], data4['areaUnderSignal'].loc[index2:], c='b')

ax.scatter(data4['RMS'].loc[index1:index2], data4['Stv'].loc[index1:index2], data4['areaUnderSignal'].loc[index1:index2], c='r', marker='o', label='Anomaly data point')
ax.set_xlabel('RMS')
ax.set_ylabel('Stv')
ax.set_zlabel('areaUnderSignal')
ax.legend()
ax.set_title('Failure Case 6')
plt.show()


# # Local Outlier Factor

# In[180]:


data1 = feature_axis1_FC1
data2 = feature_axis2_FC1
data3 = feature_axis3_FC1
data4 = feature_axis4_FC1
data5 = feature_axis5_FC1
data6 = feature_axis6_FC1
frames_FC1 = [data1, data2, data3, data4, data5, data6]
OneclassSVM_global_FC1 = OneclassSVM(frames_FC1,10,'global novelty detection OneclassSVM FC1',line_FC1)


# In[181]:


data1 = feature_axis1_FC6
data2 = feature_axis2_FC6
data3 = feature_axis3_FC6
data4 = feature_axis4_FC6
data5 = feature_axis5_FC6
data6 = feature_axis6_FC6
frames_FC6 = [data1, data2, data3, data4, data5, data6]
OneclassSVM_global_FC6 = OneclassSVM(frames_FC6,10,'global novelty detection OneclassSVM FC1',line_FC6)


# # Autoencoder

# In[115]:


data1 = feature_axis1_FC1
data2 = feature_axis2_FC1
data3 = feature_axis3_FC1
data4 = feature_axis4_FC1
data5 = feature_axis5_FC1
data6 = feature_axis6_FC1
frames = [data1, data2, data3, data4, data5, data6]
autoencoder(frames,'Autoencoder FC1',20,line_FC1)


# In[116]:


data1 = feature_axis1_FC6
data2 = feature_axis2_FC6
data3 = feature_axis3_FC6
data4 = feature_axis4_FC6
data5 = feature_axis5_FC6
data6 = feature_axis6_FC6
frames = [data1, data2, data3, data4, data5, data6]
autoencoder(frames,'Autoencoder FC6',20,line_FC6)


# In[182]:


feature_axis1_FC7 = pd.concat([RMS_Value_FC7['axis1'],Stv_FC7['axis1'],torque_areaUnderSignal_FC7['axis1']],axis=1)
feature_axis1_FC7.columns = ['RMS','Stv','areaUnderSignal']

feature_axis1_FC8 = pd.concat([RMS_Value_FC8['axis1'],Stv_FC8['axis1'],torque_areaUnderSignal_FC8['axis1']],axis=1)
feature_axis1_FC8.columns = ['RMS','Stv','areaUnderSignal']

feature_axis2_FC7 = pd.concat([RMS_Value_FC7['axis2'],Stv_FC7['axis2'],torque_areaUnderSignal_FC7['axis2']],axis=1)
feature_axis2_FC7.columns = ['RMS','Stv','areaUnderSignal']

feature_axis2_FC8 = pd.concat([RMS_Value_FC8['axis2'],Stv_FC8['axis2'],torque_areaUnderSignal_FC8['axis2']],axis=1)
feature_axis2_FC8.columns = ['RMS','Stv','areaUnderSignal']

feature_axis3_FC7 = pd.concat([RMS_Value_FC7['axis3'],Stv_FC7['axis3'],torque_areaUnderSignal_FC7['axis3']],axis=1)
feature_axis3_FC7.columns = ['RMS','Stv','areaUnderSignal']

feature_axis3_FC8 = pd.concat([RMS_Value_FC8['axis3'],Stv_FC8['axis3'],torque_areaUnderSignal_FC8['axis3']],axis=1)
feature_axis3_FC8.columns = ['RMS','Stv','areaUnderSignal']

feature_axis4_FC7 = pd.concat([RMS_Value_FC7['axis4'],Stv_FC7['axis4'],torque_areaUnderSignal_FC7['axis4']],axis=1)
feature_axis4_FC7.columns = ['RMS','Stv','areaUnderSignal']

feature_axis4_FC8 = pd.concat([RMS_Value_FC8['axis4'],Stv_FC8['axis4'],torque_areaUnderSignal_FC8['axis4']],axis=1)
feature_axis4_FC8.columns = ['RMS','Stv','areaUnderSignal']

feature_axis5_FC7 = pd.concat([RMS_Value_FC7['axis5'],Stv_FC7['axis5'],torque_areaUnderSignal_FC7['axis5']],axis=1)
feature_axis5_FC7.columns = ['RMS','Stv','areaUnderSignal']

feature_axis5_FC8 = pd.concat([RMS_Value_FC8['axis5'],Stv_FC8['axis5'],torque_areaUnderSignal_FC8['axis5']],axis=1)
feature_axis5_FC8.columns = ['RMS','Stv','areaUnderSignal']

feature_axis6_FC7 = pd.concat([RMS_Value_FC7['axis6'],Stv_FC7['axis6'],torque_areaUnderSignal_FC7['axis6']],axis=1)
feature_axis6_FC7.columns = ['RMS','Stv','areaUnderSignal']

feature_axis6_FC8 = pd.concat([RMS_Value_FC8['axis6'],Stv_FC8['axis6'],torque_areaUnderSignal_FC8['axis6']],axis=1)
feature_axis6_FC8.columns = ['RMS','Stv','areaUnderSignal']


# In[ ]:


line_FC7 = [date(2017, 4, 20),date(2017, 4, 26)]
line_FC8 = [date(2016, 10, 19),date(2017, 1, 23)]


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
index1 = date(2017, 4, 20)
index2 = date(2017, 4, 26)
fig = plt.figure() 
ax = fig.add_subplot(projection='3d') 

data1 = feature_axis1_FC7
data2 = feature_axis2_FC7
data3 = feature_axis3_FC7
data4 = feature_axis4_FC7
data5 = feature_axis5_FC7
data6 = feature_axis6_FC7

ax.scatter(data1['RMS'], data1['Stv'], data1['areaUnderSignal'], c='g',marker = 'o')

ax.scatter(data3['RMS'], data3['Stv'], data3['areaUnderSignal'], c='y',marker = 'v')
ax.scatter(data4['RMS'], data4['Stv'], data4['areaUnderSignal'], c='k',marker = 's')
ax.scatter(data5['RMS'], data5['Stv'], data5['areaUnderSignal'], c='m', marker = 'p')
ax.scatter(data6['RMS'], data6['Stv'], data6['areaUnderSignal'], c='c',marker = '*')
# ax.scatter(data2['RMS'], data2['Stv'], data2['areaUnderSignal'], c='b')
ax.scatter(data2['RMS'].loc[:index1], data2['Stv'].loc[:index1], data2['areaUnderSignal'].loc[:index1], alpha =0.2, c='b')
ax.scatter(data2['RMS'].loc[index2:], data2['Stv'].loc[index2:], data2['areaUnderSignal'].loc[index2:], alpha =0.2, c='b')

ax.scatter(data2['RMS'].loc[index1:index2], data2['Stv'].loc[index1:index2], data2['areaUnderSignal'].loc[index1:index2], c='r', marker='o', label='Anomaly data point')
ax.set_xlabel('RMS')
ax.set_ylabel('Stv')
ax.set_zlabel('areaUnderSignal')
ax.legend()
ax.set_title('Failure Case 7')
plt.show()


# In[ ]:


fig = plt.figure() 
ax = fig.add_subplot(projection='3d')

Index = pd.to_datetime(feature_axis1_FC7.index)
Markers = ['x','v','s','p','o','*']
axis_list = ['axis1','axis2','axis3','axis4','axis5','axis6']
i = 0
for data in frames_FC7:
    sca = ax.scatter(data['RMS'], data['Stv'], data['areaUnderSignal'], c=Index, marker = Markers[i],cmap='jet_r', label = axis_list[i])
    i = i + 1

cbar = plt.colorbar(sca,shrink=0.7, pad= 0.1)

cbar.ax.set_yticklabels(pd.to_datetime(cbar.get_ticks()).strftime(date_format='%b %Y'))
ax.set_xlabel('RMS') 
ax.set_ylabel('Stv') 
ax.set_zlabel('areaUnderSignal') 
ax.legend()
ax.set_title('Failure Case 7')
plt.show()


# In[ ]:


data1 = feature_axis1_FC7
data2 = feature_axis2_FC7
data3 = feature_axis3_FC7
data4 = feature_axis4_FC7
data5 = feature_axis5_FC7
data6 = feature_axis6_FC7
frames_FC7 = [data1, data2, data3, data4, data5, data6]

fig, (ax1, ax2) = plt.subplots(2,figsize=(18,10))
Distance_FC7_Global = plot_distance_global(frames_FC7,10,'FC7 Centroid-based global',ax1,line_FC7)

Distance_FC7_local = plot_distance_local(frames_FC7,10, 'FC7 Centroid-based local',ax2,line_FC7)


# In[ ]:


fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(18,10))
k = 10
Mean = []
std = []
columns=['axis1','axis2','axis3','axis4','axis5','axis6']
Distance_Global_FC7 =pd.DataFrame(columns=columns)
for i, (fc, d) in enumerate(Distance_FC7_Global.items()):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    x = pd.to_datetime(data1[k+1:].index)
    Distance_Global_FC7[fc] = d
    Average = Distance_Global_FC7[fc].rolling(7).mean()
    Moved_average = np.mean(d) + np.std(d)*0.6
    ax.axhline(y = Moved_average, color = 'r', linestyle = 'dashed',label= 'Moved average (Multiplier = 0.6)')
    ax.plot(x, Average,label = fc)
    if fc == 'axis2':
        ax.set_title(f'Failure Case 7 {fc}',color = 'r')
    else:
        ax.set_title(f'Failure Case 7 {fc}')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Centroid-based Value')
    ax.set_ylim(0, .5)
    ax.legend(loc= "best",fontsize="6")
    
plt.tight_layout()


# In[ ]:


data1 = feature_axis1_FC7
data2 = feature_axis2_FC7
data3 = feature_axis3_FC7
data4 = feature_axis4_FC7
data5 = feature_axis5_FC7
data6 = feature_axis6_FC7
frames_FC7 = [data1, data2, data3, data4, data5, data6]
fig, (ax1, ax2) = plt.subplots(2,figsize=(18,10))
Distance_based_lof_global(frames_FC7,10,'global distance_based LOF FC7',ax1,line_FC7)
Distance_based_lof_local(frames_FC7,10,'local distance_based LOF FC7',ax2,line_FC7)


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
index1 = date(2016, 10, 9)
index2 = date(2017, 1, 23)
fig = plt.figure() 
ax = fig.add_subplot(projection='3d') 

data1 = feature_axis1_FC8
data2 = feature_axis2_FC8
data3 = feature_axis3_FC8
data4 = feature_axis4_FC8
data5 = feature_axis5_FC8
data6 = feature_axis6_FC8
ax.scatter(data1['RMS'], data1['Stv'], data1['areaUnderSignal'], c='g',marker = 'o')


ax.scatter(data2['RMS'], data2['Stv'], data2['areaUnderSignal'], c='k',marker = 's')
ax.scatter(data4['RMS'], data4['Stv'], data4['areaUnderSignal'], c='y',marker = 'v')
ax.scatter(data5['RMS'], data5['Stv'], data5['areaUnderSignal'], c='m', marker = 'p')
ax.scatter(data6['RMS'], data6['Stv'], data6['areaUnderSignal'], c='c',marker = '*')
ax.scatter(data3['RMS'].loc[:index1], data3['Stv'].loc[:index1], data3['areaUnderSignal'].loc[:index1], c='b')
ax.scatter(data3['RMS'].loc[index2:], data3['Stv'].loc[index2:], data3['areaUnderSignal'].loc[index2:], c='b')

ax.scatter(data3['RMS'].loc[index1:index2], data3['Stv'].loc[index1:index2], data3['areaUnderSignal'].loc[index1:index2], c='r', marker='o', label='Anomaly data point')
ax.set_xlabel('RMS')
ax.set_ylabel('Stv')
ax.set_zlabel('areaUnderSignal')
ax.legend()
ax.set_title('Failure Case 8')
plt.show()


# In[ ]:


fig = plt.figure() 
ax = fig.add_subplot(projection='3d')


Index = pd.to_datetime(feature_axis1_FC8.index)
Markers = ['x','v','s','p','o','*']
axis_list = ['axis1','axis2','axis3','axis4','axis5','axis6']
i = 0
for data in frames_FC8:
    sca = ax.scatter(data['RMS'], data['Stv'], data['areaUnderSignal'], c=Index, marker = Markers[i],cmap='jet_r', label = axis_list[i])
    i = i + 1

cbar = plt.colorbar(sca, shrink=0.6, pad= 0.1)

cbar.ax.set_yticklabels(pd.to_datetime(cbar.get_ticks()).strftime(date_format='%b %Y'))
ax.set_xlabel('RMS') 
ax.set_ylabel('Stv') 
ax.set_zlabel('areaUnderSignal') 
ax.legend()
ax.set_title('Failure Case 8')
plt.show()


# In[ ]:


data1 = feature_axis1_FC8
data2 = feature_axis2_FC8
data3 = feature_axis3_FC8
data4 = feature_axis4_FC8
data5 = feature_axis5_FC8
data6 = feature_axis6_FC8
frames_FC8 = [data1, data2, data3, data4, data5, data6]

fig, (ax1, ax2) = plt.subplots(2,figsize=(18,10))
Distance_FC8_Global= plot_distance_global(frames_FC8,10,'FC8 Centroid-based global',ax1, line_FC8)

Distance_FC8_local = plot_distance_local(frames_FC8,10, 'FC8 Centroid-based local',ax2,line_FC8)


# In[ ]:


fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(18,10))
k = 10
Mean = []
std = []
columns=['axis1','axis2','axis3','axis4','axis5','axis6']
Distance_Global_FC8 =pd.DataFrame(columns=columns)
for i, (fc, d) in enumerate(Distance_FC8_Global.items()):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    x = pd.to_datetime(data1[k+1:].index)
    Distance_Global_FC8[fc] = d
    Average = Distance_Global_FC8[fc].rolling(7).mean()
    Moved_average = np.mean(d) + np.std(d)*0.6
    ax.axhline(y = Moved_average, color = 'r', linestyle = 'dashed',label= 'Moved average (Multiplier = 0.6)')
    ax.plot(x, Average,label = fc)
    if fc == 'axis3':
        ax.set_title(f'Failure Case 8 {fc}',color = 'r')
    else:
        ax.set_title(f'Failure Case 8 {fc}')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Centroid-based Value')
    ax.set_ylim(0, .5)
    ax.legend(loc= "best",fontsize="6")
    
plt.tight_layout()


# In[ ]:


data1 = feature_axis1_FC8
data2 = feature_axis2_FC8
data3 = feature_axis3_FC8
data4 = feature_axis4_FC8
data5 = feature_axis5_FC8
data6 = feature_axis6_FC8
frames_FC8 = [data1, data2, data3, data4, data5, data6]
fig, (ax1, ax2) = plt.subplots(2,figsize=(18,10))
Distance_based_lof_global(frames_FC8,10,'global distance_based LOF FC8',ax1,line_FC8)
Distance_based_lof_local(frames_FC8,10,'local distance_based LOF FC8',ax2,line_FC8)


# In[183]:


data1 = feature_axis1_FC7
data2 = feature_axis2_FC7
data3 = feature_axis3_FC7
data4 = feature_axis4_FC7
data5 = feature_axis5_FC7
data6 = feature_axis6_FC7
frames_FC7 = [data1, data2, data3, data4, data5, data6]
OneclassSVM_global_FC7 = OneclassSVM(frames_FC7,10,'global novelty detection OneclassSVM FC7',line_FC7)


# In[184]:


data1 = feature_axis1_FC8
data2 = feature_axis2_FC8
data3 = feature_axis3_FC8
data4 = feature_axis4_FC8
data5 = feature_axis5_FC8
data6 = feature_axis6_FC8
frames_FC8 = [data1, data2, data3, data4, data5, data6]
OneclassSVM_global_FC6 = OneclassSVM(frames_FC8,10,'global novelty detection OneclassSVM FC8',line_FC8)


# # Autoencoder

# In[ ]:


data1 = feature_axis1_FC7
data2 = feature_axis2_FC7
data3 = feature_axis3_FC7
data4 = feature_axis4_FC7
data5 = feature_axis5_FC7
data6 = feature_axis6_FC7
frames = [data1, data2, data3, data4, data5, data6]
frames = [data1, data2, data3, data4, data5, data6]
autoencoder(frames,'Autoencoder FC7',20,line_FC7)


# In[ ]:


data1 = feature_axis1_FC8
data2 = feature_axis2_FC8
data3 = feature_axis3_FC8
data4 = feature_axis4_FC8
data5 = feature_axis5_FC8
data6 = feature_axis6_FC8
frames = [data1, data2, data3, data4, data5, data6]
autoencoder(frames,'Autoencoder FC8',20,line_FC8)


# In[ ]:


feature_axis1_FC11 = pd.concat([RMS_Value_FC11['axis1'],Stv_FC11['axis1'],torque_areaUnderSignal_FC11['axis1']],axis=1)
feature_axis1_FC11.columns = ['RMS','Stv','areaUnderSignal']

feature_axis1_FC12 = pd.concat([RMS_Value_FC12['axis1'],Stv_FC12['axis1'],torque_areaUnderSignal_FC12['axis1']],axis=1)
feature_axis1_FC12.columns = ['RMS','Stv','areaUnderSignal']

feature_axis2_FC11 = pd.concat([RMS_Value_FC11['axis2'],Stv_FC11['axis2'],torque_areaUnderSignal_FC11['axis2']],axis=1)
feature_axis2_FC11.columns = ['RMS','Stv','areaUnderSignal']

feature_axis2_FC12 = pd.concat([RMS_Value_FC12['axis2'],Stv_FC12['axis2'],torque_areaUnderSignal_FC12['axis2']],axis=1)
feature_axis2_FC12.columns = ['RMS','Stv','areaUnderSignal']

feature_axis3_FC11 = pd.concat([RMS_Value_FC11['axis3'],Stv_FC11['axis3'],torque_areaUnderSignal_FC11['axis3']],axis=1)
feature_axis3_FC11.columns = ['RMS','Stv','areaUnderSignal']

feature_axis3_FC12 = pd.concat([RMS_Value_FC12['axis3'],Stv_FC12['axis3'],torque_areaUnderSignal_FC12['axis3']],axis=1)
feature_axis3_FC12.columns = ['RMS','Stv','areaUnderSignal']

feature_axis4_FC11 = pd.concat([RMS_Value_FC11['axis4'],Stv_FC11['axis4'],torque_areaUnderSignal_FC11['axis4']],axis=1)
feature_axis4_FC11.columns = ['RMS','Stv','areaUnderSignal']

feature_axis4_FC12 = pd.concat([RMS_Value_FC12['axis4'],Stv_FC12['axis4'],torque_areaUnderSignal_FC12['axis4']],axis=1)
feature_axis4_FC12.columns = ['RMS','Stv','areaUnderSignal']

feature_axis5_FC11 = pd.concat([RMS_Value_FC11['axis5'],Stv_FC11['axis5'],torque_areaUnderSignal_FC11['axis5']],axis=1)
feature_axis5_FC11.columns = ['RMS','Stv','areaUnderSignal']

feature_axis5_FC12 = pd.concat([RMS_Value_FC12['axis6'],Stv_FC12['axis5'],torque_areaUnderSignal_FC12['axis5']],axis=1)
feature_axis5_FC12.columns = ['RMS','Stv','areaUnderSignal']

feature_axis6_FC11 = pd.concat([RMS_Value_FC11['axis6'],Stv_FC11['axis6'],torque_areaUnderSignal_FC11['axis6']],axis=1)
feature_axis6_FC11.columns = ['RMS','Stv','areaUnderSignal']

feature_axis6_FC12 = pd.concat([RMS_Value_FC12['axis6'],Stv_FC12['axis6'],torque_areaUnderSignal_FC12['axis6']],axis=1)
feature_axis6_FC12.columns = ['RMS','Stv','areaUnderSignal']


# In[ ]:


line_FC11 = [[date(2017, 11, 26),date(2017, 12, 3)],
             [date(2017, 12, 3),date(2017, 12, 6)]]

line_FC12 = [[date(2017, 12, 17),date(2017, 12, 22)],
            [date(2017,4,22), date(2017,4,27)]]


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
index3 = date(2017, 12, 3)
index2 = date(2017, 12, 6)
index1 = date(2017, 11, 26)
fig = plt.figure() 
ax = fig.add_subplot(projection='3d') 

data1 = feature_axis1_FC11
data2 = feature_axis2_FC11
data3 = feature_axis3_FC11
data4 = feature_axis4_FC11
data5 = feature_axis5_FC11
data6 = feature_axis6_FC11
ax.scatter(data1['RMS'], data1['Stv'], data1['areaUnderSignal'], c='g',marker = 'p')


ax.scatter(data3['RMS'], data3['Stv'], data3['areaUnderSignal'], c='k',marker = 's')
ax.scatter(data4['RMS'], data4['Stv'], data4['areaUnderSignal'], c='y',marker = 'v')
ax.scatter(data6['RMS'], data6['Stv'], data6['areaUnderSignal'], c='c',marker = '*')

ax.scatter(data5['RMS'].loc[:index3], data5['Stv'].loc[:index3], data5['areaUnderSignal'].loc[:index3], c='m', marker = 'x')



ax.scatter(data2['RMS'].loc[:index1], data2['Stv'].loc[:index1], data2['areaUnderSignal'].loc[:index1],c='b')


ax.scatter(data2['RMS'].loc[index1:index2], data2['Stv'].loc[index1:index2], data2['areaUnderSignal'].loc[index1:index2], c='r', marker='o', label='Anomaly data point')
ax.scatter(data5['RMS'].loc[index3:], data5['Stv'].loc[index3:], data5['areaUnderSignal'].loc[index3:], c='r', marker='o')

ax.set_xlabel('RMS')
ax.set_ylabel('Stv')
ax.set_zlabel('areaUnderSignal')
ax.legend()
ax.set_title('Failure Case 11')
plt.show()


# In[ ]:


fig = plt.figure() 
ax = fig.add_subplot(projection='3d')


Index = pd.to_datetime(feature_axis1_FC11.index)
Markers = ['x','v','s','p','o','*']
axis_list = ['axis1','axis2','axis3','axis4','axis5','axis6']
i = 0
for data in frames_FC11:
    sca = ax.scatter(data['RMS'], data['Stv'], data['areaUnderSignal'], c=Index, marker = Markers[i],cmap='jet_r', label = axis_list[i])
    i = i + 1

cbar = plt.colorbar(sca, shrink=0.6, pad= 0.1)

cbar.ax.set_yticklabels(pd.to_datetime(cbar.get_ticks()).strftime(date_format='%b %Y'))
ax.set_xlabel('RMS') 
ax.set_ylabel('Stv') 
ax.set_zlabel('areaUnderSignal') 
ax.legend()
ax.set_title('Failure Case 11')
plt.show()


# In[ ]:


data1 = feature_axis1_FC11
data2 = feature_axis2_FC11
data3 = feature_axis3_FC11
data4 = feature_axis4_FC11
data5 = feature_axis5_FC11
data6 = feature_axis6_FC11
frames_FC11 = [data1, data2, data3, data4, data5, data6]
k = len(data1)
result = pd.concat(frames_FC11)
from pyod.models.lof import LOF

#Local Outlier Factor
clf_name = 'LOF'
clf = LOF()
clf.fit(result)

# get the prediction labels and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
IOF_scores = clf.decision_scores_  # raw outlier scores

fig = plt.figure(figsize=(15, 15))
ax1 = fig.add_subplot(1, 3, 1, projection='3d')

sca1 = ax1.scatter(result['RMS'], result['Stv'], result['areaUnderSignal'],
c=IOF_scores, cmap='seismic', s=50)

ax1.set_xlabel('RMS')
ax1.set_ylabel('Stv')
ax1.set_zlabel('areaUnderSignal')
ax1.set_title('Failure_case 11 LOF')
plt.colorbar(sca1, shrink=0.3, pad= 0.1)

#KNN
clf_name = 'KNN'
clf = KNN()
clf.fit(result)

# get the prediction label and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
KNN_scores = clf.decision_scores_  # raw outlier scores



ax2 = fig.add_subplot(1, 3, 2, projection='3d')
sca2 = ax2.scatter(result['RMS'], result['Stv'], result['areaUnderSignal'],
c=KNN_scores, cmap='seismic', s=50)

ax2.set_xlabel('RMS')
ax2.set_ylabel('Stv')
ax2.set_zlabel('areaUnderSignal')
ax2.set_title('Failure_case 11 KNN')
plt.colorbar(sca2, shrink=0.3, pad= 0.1)

#Anomaly 

index3 = date(2017, 12, 3)
index2 = date(2017, 12, 6)
index1 = date(2017, 11, 26)
fig = plt.figure(figsize=(15, 15))
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.scatter(data1['RMS'], data1['Stv'], data1['areaUnderSignal'], c='b')


ax3.scatter(data3['RMS'], data3['Stv'], data3['areaUnderSignal'], c='b')
ax3.scatter(data4['RMS'], data4['Stv'], data4['areaUnderSignal'], c='b')
ax3.scatter(data6['RMS'], data6['Stv'], data6['areaUnderSignal'], c='b')

ax3.scatter(data5['RMS'].loc[:index3], data5['Stv'].loc[:index3], data5['areaUnderSignal'].loc[:index3], c='b')



ax3.scatter(data2['RMS'].loc[:index1], data2['Stv'].loc[:index1], data2['areaUnderSignal'].loc[:index1],c='b')


ax3.scatter(data2['RMS'].loc[index1:index2], data2['Stv'].loc[index1:index2], data2['areaUnderSignal'].loc[index1:index2], c='r', marker='o', label='Anomaly data point')
ax3.scatter(data5['RMS'].loc[index3:], data5['Stv'].loc[index3:], data5['areaUnderSignal'].loc[index3:], c='r', marker='o')

ax3.set_xlabel('RMS')
ax3.set_ylabel('Stv')
ax3.set_zlabel('areaUnderSignal')
ax3.legend()
ax3.set_title('Failure Case 11')
plt.show()


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D


index1 = date(2017, 4, 22)
index2 = date(2017, 4, 27)

index3 = date(2017, 12, 17)
index4 = date(2017, 12, 23)

fig = plt.figure() 
ax = fig.add_subplot(projection='3d') 

data1 = feature_axis1_FC12
data2 = feature_axis2_FC12
data3 = feature_axis3_FC12
data4 = feature_axis4_FC12
data5 = feature_axis5_FC12
data6 = feature_axis6_FC12
ax.scatter(data1['RMS'], data1['Stv'], data1['areaUnderSignal'], c='g',marker = 'p')


ax.scatter(data3['RMS'], data3['Stv'], data3['areaUnderSignal'], c='k',marker = 's')
ax.scatter(data4['RMS'], data4['Stv'], data4['areaUnderSignal'], c='y',marker = 'v')
ax.scatter(data6['RMS'], data6['Stv'], data6['areaUnderSignal'], c='c',marker = '*')

ax.scatter(data5['RMS'].loc[:index3], data5['Stv'].loc[:index3], data5['areaUnderSignal'].loc[:index3], c='m', marker = 'x')
ax.scatter(data5['RMS'].loc[index4:], data5['Stv'].loc[index4:], data5['areaUnderSignal'].loc[index4:], c='m', marker = 'x')


ax.scatter(data2['RMS'].loc[:index1], data2['Stv'].loc[:index1], data2['areaUnderSignal'].loc[:index1],c='b')
ax.scatter(data2['RMS'].loc[index2:], data2['Stv'].loc[index2:], data2['areaUnderSignal'].loc[index2:],c='b')

ax.scatter(data2['RMS'].loc[index1:index2], data2['Stv'].loc[index1:index2], data2['areaUnderSignal'].loc[index1:index2], c='r', marker='o', label='Anomaly data point')
ax.scatter(data5['RMS'].loc[index3:index4], data5['Stv'].loc[index3:index4], data5['areaUnderSignal'].loc[index3:index4], c='r', marker='o')

ax.set_xlabel('RMS')
ax.set_ylabel('Stv')
ax.set_zlabel('areaUnderSignal')
ax.legend()
ax.set_title('Failure Case 12')
plt.show()


# In[ ]:


fig = plt.figure() 
ax = fig.add_subplot(projection='3d')


Index = pd.to_datetime(feature_axis1_FC12.index)
Markers = ['x','v','s','p','o','*']
axis_list = ['axis1','axis2','axis3','axis4','axis5','axis6']
i = 0
for data in frames_FC12:
    sca = ax.scatter(data['RMS'], data['Stv'], data['areaUnderSignal'], c=Index, marker = Markers[i],cmap='jet_r', label = axis_list[i])
    i = i + 1

cbar = plt.colorbar(sca, shrink=0.6, pad= 0.1)

cbar.ax.set_yticklabels(pd.to_datetime(cbar.get_ticks()).strftime(date_format='%b %Y'))
ax.set_xlabel('RMS') 
ax.set_ylabel('Stv') 
ax.set_zlabel('areaUnderSignal') 
ax.legend()
ax.set_title('Failure Case 12')
plt.show()


# In[ ]:


data1 = feature_axis1_FC11
data2 = feature_axis2_FC11
data3 = feature_axis3_FC11
data4 = feature_axis4_FC11
data5 = feature_axis5_FC11
data6 = feature_axis6_FC11
frames_FC11 = [data1, data2, data3, data4, data5, data6]

fig, (ax1, ax2) = plt.subplots(2,figsize=(18,10))
Distance_FC11_Global = plot_distance_global(frames_FC11,10,'FC11 Centroid-based global',ax1,line_FC11)

Distance_FC11_local = plot_distance_local(frames_FC11,10, 'FC11 Centroid-based local',ax2,line_FC11)


# In[ ]:


fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(18,10))
k = 10
Mean = []
std = []
columns=['axis1','axis2','axis3','axis4','axis5','axis6']
Distance_Global_FC11 =pd.DataFrame(columns=columns)
for i, (fc, d) in enumerate(Distance_FC11_Global.items()):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    x = pd.to_datetime(data1[k+1:].index)
    Distance_Global_FC11[fc] = d
    Average = Distance_Global_FC11[fc].rolling(7).mean()
    Moved_average = np.mean(d) + np.std(d)*0.6
    ax.axhline(y = Moved_average, color = 'r', linestyle = 'dashed',label= 'Moved average (Multiplier = 0.6)')
    ax.plot(x, Average,label = fc)
    if fc == 'axis2' or fc =='axis5':
        ax.set_title(f'Failure Case 11 {fc}',color = 'r')
    else:
        ax.set_title(f'Failure Case 11 {fc}')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Centroid-based Value')
    ax.set_ylim(0, .5)
    ax.legend(loc= "best",fontsize="6")
    
plt.tight_layout()


# In[ ]:


data1 = feature_axis1_FC12
data2 = feature_axis2_FC12
data3 = feature_axis3_FC12
data4 = feature_axis4_FC12
data5 = feature_axis5_FC12
data6 = feature_axis6_FC12
frames_FC12 = [data1, data2, data3, data4, data5, data6]

fig, (ax1, ax2) = plt.subplots(2,figsize=(18,10))
Distance_FC12_Global = plot_distance_global(frames_FC12,10,'FC12 Centroid-based global',ax1, line_FC12)

Distance_FC12_local = plot_distance_local(frames_FC12,10, 'FC12 Centroid-based local',ax2, line_FC12)


# In[ ]:


fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(18,10))
k = 10
Mean = []
std = []
columns=['axis1','axis2','axis3','axis4','axis5','axis6']
Distance_Global_FC12 =pd.DataFrame(columns=columns)
for i, (fc, d) in enumerate(Distance_FC12_Global.items()):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    x = pd.to_datetime(data1[k+1:].index)
    Distance_Global_FC12[fc] = d
    Average = Distance_Global_FC12[fc].rolling(7).mean()
    Moved_average = np.mean(d) + np.std(d)*0.6
    ax.axhline(y = Moved_average, color = 'r', linestyle = 'dashed',label= 'Moved average (Multiplier = 0.6)')
    ax.plot(x, Average,label = fc)
    if fc == 'axis2' or fc == 'axis5':
        ax.set_title(f'Failure Case 12 {fc}',color = 'r')
    else:
        ax.set_title(f'Failure Case 12 {fc}')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Centroid-based Value')
    ax.set_ylim(0, .5)
    ax.legend(loc= "best",fontsize="6")
    
plt.tight_layout()


# In[ ]:


data1 = feature_axis1_FC11
data2 = feature_axis2_FC11
data3 = feature_axis3_FC11
data4 = feature_axis4_FC11
data5 = feature_axis5_FC11
data6 = feature_axis6_FC11
frames_FC11 = [data1, data2, data3, data4, data5, data6]
fig, (ax1, ax2) = plt.subplots(2,figsize=(18,10))
Distance_based_lof_global(frames_FC11,10,'global distance_based LOF FC11',ax1,line_FC11)
Distance_based_lof_local(frames_FC11,10,'local distance_based LOF FC11',ax2,line_FC11)


# In[ ]:


data1 = feature_axis1_FC12
data2 = feature_axis2_FC12
data3 = feature_axis3_FC12
data4 = feature_axis4_FC12
data5 = feature_axis5_FC12
data6 = feature_axis6_FC12
frames_FC12 = [data1, data2, data3, data4, data5, data6]
fig, (ax1, ax2) = plt.subplots(2,figsize=(18,10))
Distance_based_lof_global(frames_FC12,10,'global distance_based LOF FC12',ax1,line_FC12)
Distance_based_lof_local(frames_FC12,10,'local distance_based LOF FC12',ax2,line_FC12)


# In[ ]:


data1 = feature_axis1_FC12
data2 = feature_axis2_FC12
data3 = feature_axis3_FC12
data4 = feature_axis4_FC12
data5 = feature_axis5_FC12
data6 = feature_axis6_FC12
frames_FC12 = [data1, data2, data3, data4, data5, data6]
k = len(data1)
result = pd.concat(frames_FC12)
from pyod.models.lof import LOF

#Local Outlier Factor
clf_name = 'LOF'
clf = LOF()
clf.fit(result)

# get the prediction labels and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
IOF_scores = clf.decision_scores_  # raw outlier scores

fig = plt.figure(figsize=(15, 15))
ax1 = fig.add_subplot(1, 3, 1, projection='3d')

sca1 = ax1.scatter(result['RMS'], result['Stv'], result['areaUnderSignal'],
c=IOF_scores, cmap='seismic', s=50)

ax1.set_xlabel('RMS')
ax1.set_ylabel('Stv')
ax1.set_zlabel('areaUnderSignal')
ax1.set_title('Failure_case 12 LOF')
plt.colorbar(sca1, shrink=0.3, pad= 0.1)

#KNN
clf_name = 'KNN'
clf = KNN()
clf.fit(result)

# get the prediction label and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
KNN_scores = clf.decision_scores_  # raw outlier scores



ax2 = fig.add_subplot(1, 3, 2, projection='3d')
sca2 = ax2.scatter(result['RMS'], result['Stv'], result['areaUnderSignal'],
c=KNN_scores, cmap='seismic', s=50)

ax2.set_xlabel('RMS')
ax2.set_ylabel('Stv')
ax2.set_zlabel('areaUnderSignal')
ax2.set_title('Failure_case 12 KNN')
plt.colorbar(sca2, shrink=0.3, pad= 0.1)


fig = plt.figure(figsize=(15, 15))
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.scatter(data1['RMS'], data1['Stv'], data1['areaUnderSignal'], c='b')


ax3.scatter(data3['RMS'], data3['Stv'], data3['areaUnderSignal'],  c='b')
ax3.scatter(data4['RMS'], data4['Stv'], data4['areaUnderSignal'],  c='b')
ax3.scatter(data6['RMS'], data6['Stv'], data6['areaUnderSignal'],  c='b')

ax3.scatter(data5['RMS'].loc[:index3], data5['Stv'].loc[:index3], data5['areaUnderSignal'].loc[:index3], c='b')
ax3.scatter(data5['RMS'].loc[index4:], data5['Stv'].loc[index4:], data5['areaUnderSignal'].loc[index4:],  c='b')


ax3.scatter(data2['RMS'].loc[:index1], data2['Stv'].loc[:index1], data2['areaUnderSignal'].loc[:index1],c='b')
ax3.scatter(data2['RMS'].loc[index2:], data2['Stv'].loc[index2:], data2['areaUnderSignal'].loc[index2:],c='b')

ax3.scatter(data2['RMS'].loc[index1:index2], data2['Stv'].loc[index1:index2], data2['areaUnderSignal'].loc[index1:index2], c='r', marker='o', label='Anomaly data point')
ax3.scatter(data5['RMS'].loc[index3:index4], data5['Stv'].loc[index3:index4], data5['areaUnderSignal'].loc[index3:index4], c='r', marker='o')

ax3.set_xlabel('RMS')
ax3.set_ylabel('Stv')
ax3.set_zlabel('areaUnderSignal')
ax3.legend()
ax3.set_title('Failure Case 12')
plt.show()

