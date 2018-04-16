"""
The Waveform class creates, manipulate and plot discrete functions of time.

It was primarily designed to work with the analog input/output functions
of the MCCDAQ Python driver :mod:`usb2600.usb2600`, but is much more general in scope.

The :class:`Waveform` class can do:
  - single- and multi-channel waveforms
  - time plots
  - XY plots
  - XYZ plots (both regularly and randomly spaced data)
  - Fourier transform
  - simple arithmetics (add and multiply waveforms, and waveforms and scalars)
  
Functions are included to create some usual waveforms:
  - sine
  - helix
  - gosine (connects two points with a sine curve)
  - linscan (triangular waveform with rounded edges)
  
Copyright Guillaume Lepert, 2014
"""

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import math

twopi = 2*math.pi

#: Define time units, relative to the second.
timescale = {'s':1., 'ms':1000., 'us':1.0E6, 'ns': 1.0E9,
       'min':1/60., 'h':1/3600., 'day': 86400., 'week': 604800, 'year': 3.15576E7}

class Waveform:
  """A simple class to create and manipulate waveforms (functions of time). """
  def __init__(self, data, dt, t0=0, padding=1):
    """
    :param data: the data
    :type  data: array-like
    :param float dt: the waveform sampling interval
    :param float t0: the waveform time origin (timestamp of first element of ``data``)
    :param int padding: minimum length of the waveform, which will be padded right with 0 if its length is smaller.
    """
    #pad = data.shape[1] % padding
    #if pad > 0: 
    #  data = np.hstack(data, np.zeros(shape=(data.shape[0], padding - pad), dtype=data.dtype))
    self.data = data
    self.dt = float(dt)
    self.t0 = float(t0)
    self.rate = 1/self.dt
    if data.ndim == 1:    # reshape 1D array to 2D
      self.data = self.data.reshape((1, len(data)))
    shape = self.data.shape
    self.nchannels = shape[0]
    self.nsamples = shape[1]
    self.pad(padding)
    
  def pad(self, padding, value=0):
    """Right padding.
    
    :param padding: the waveform final length
    :param value: what to pad with
    """
    pad = self.nsamples % padding
    if pad > 0: 
      self.data = np.hstack((self.data, 
                            np.zeros(shape=(self.nchannels, padding - pad), 
                                     dtype=self.data.dtype)))
    
  def __str__(self):
    return "Waveform: "+ str(self.nchannels) + " channels, "+ str(self.nsamples) + " samples at " + str(1/self.dt)+" Hz."
  
  def __mul__(self, other):
    if isinstance(other, Waveform):
      assert self.dt == other.dt
      assert self.nchannels == other.nchannels
      assert self.nsamples == other.nsamples
      return Waveform(other.data*self.data, self.dt, self.t0)
    else:
      return Waveform(other * self.data, self.dt, self.t0)
  
  def __add__(self, other):
    if isinstance(other, Waveform):
      assert self.dt == other.dt
      assert self.nchannels == other.nchannels
      assert self.nsamples == other.nsamples
      return Waveform(other.data + self.data, self.dt, self.t0)
    else:
      return Waveform(other + self.data, self.dt, self.t0)
  
  def __len__(self):
    return self.data.shape[1]
  
  def plot(self, style = ''):
    """Plot all waveforms vs time """
    for i in range(self.nchannels):
      plt.plot(self.t(), self.data[i, :], style)
    plt.show()

  def plotxy(self, style = '', channels=(0,1)):
    """XY plot for 2D waveforms. """
    fig = plt.plot(self.data[channels[0], :], self.data[channels[1], :], style)
    plt.show()
    return fig
  
  def plotxyz(self, res, channels=(0,1,2), interp=u'nn'):
    """Density plot of irregularly spaced data"""
    xi = np.linspace(min(self.data[channels[0]]), max(self.data[channels[0]]), res)
    yi = np.linspace(min(self.data[channels[1]]), max(self.data[channels[1]]), res)
    z=mlab.griddata(self.data[channels[0]], 
                    self.data[channels[1]], 
                    self.data[channels[2]],
                    xi, yi)
    fig = plt.figure()
    plt.pcolor(xi, yi, z)
    plt.colorbar()
    plt.show()
    return fig
  
  def plot_linear_subset(self, ch=(0,1,2), **kwargs):
    """ Density plot of the linear section of the data, if defined"""
    #try:
    sub = self.linear_subset
    n = len(sub)/4  # no of even linear segments = no of periods
    m = sub[1]-sub[0] # no of samples in linear segment
    print m
    x = np.zeros(shape=m, dtype=float)
    y = np.zeros(shape=n, dtype=float)
    Z = np.empty(shape=(n,m), dtype=float)
    for i in xrange(n):
      x += self.data[ch[0], sub[4*i]:sub[4*i+1]]  # only get every other segment.
      y[i] = np.mean(self.data[ch[1], sub[4*i]:sub[4*i+1]])
      Z[i,:] = self.data[ch[2], sub[4*i]:sub[4*i+1]]
    (X, Y) = np.meshgrid(x/(n/2), y)
    #print (X.shape, Y.shape, Z.shape)
    fig = plt.figure()
    plt.pcolormesh(X, Y, Z, **kwargs)
    plt.colorbar()
    plt.show()
    self.C = Z
    return fig
    #except AttributeError:
    #  raise Exception("Linear subset is not defined.")
     
      
  def t(self):
    """Calculate and returns the time vector. """
    time = np.arange(self.t0, self.t0 + self.dt * self.nsamples, self.dt)
    self.t = time
    return time
  
  def fourier(self, power=True, shift=False):
    """Calculate the Fourier transform of all waveforms in the object.
    
    Returns a new waveform object where dt is the frequency interval."""
    #ft = []
    #for i in range(self.nchannels):
    #  ft.append(np.fft.fft(self.amaplitude[i, :]
    out = np.fft.fft(self.data)
    t0 = 0
    if power:
      out = np.square(np.abs(out))
    if shift:
      out = np.fft.fftshift(out)
      t0 = -0.5/float(self.dt)
    
    return Waveform(out, 1/float(len(self)*self.dt), t0=t0)


def sine(frequency=1, amplitude=1, offset=0, phase=0, nsamples=100, rate=10, tunit='s'):
  """Return a sine wave.
  
  :math:`y(t) = DC + A \sin(2 \pi f t + \phi)`
  
  :param frequency: frequency *f* of the sine wave
  :param amplitude: amplitude *A* of the sine wave (peak to average)
  :param phase: phase :math:`\phi` of the sine wave
  :param offset: *DC* offset
  :param nsamples: number of samples in the final waveform
  :param rate: sampling rate, in Hz
  :param tunit: time unit to use (see :data:`timescale`)
  :rtype: :class:`Waveform`
  """
  wf = np.empty(shape=nsamples, dtype=float)
  t =  np.empty(shape=nsamples, dtype=float)
  rate = float(rate)
  for i in range(nsamples):
    wf[i] = offset+amplitude*math.sin(2.*math.pi*float(i)/rate*frequency + phase)
    t[i] = float(i)/rate*timescale[tunit]
  return Waveform(wf, 1/rate)

def helixscan(amplitude, turns):
  """Returns a two-channel waveform that makes a 2D closed helix pattern.
  
  :param amplitude: scan amplitude
  :param: turns: number of turns
  
  :Example:
  
  >>> a = waveform.helixscan(1,10)
  >>> a.plotxy()
  
  produces the following scan pattern:
  
  .. figure:: /../mccdaq/utilities/img/helixscan_demo.*
     :height: 300px
  """
  data=float(amplitude)
  turns=float(turns)
  step = 100/(math.pi * turns * turns)
  nsamples = int(100 / step)
  wf = np.empty(shape=(2, 2*nsamples+2), dtype=float)
  for i in xrange(nsamples+1):
    sqrtx = math.sqrt(i*step)
    wf[0, i] = data/10*sqrtx * math.sin(turns/5*math.pi*sqrtx)
    wf[1, i] = data/10*sqrtx * math.cos(turns/5*math.pi*sqrtx)
    wf[0, 2*nsamples-i+1] = data/10*sqrtx * math.sin(turns/5*math.pi*(100 - sqrtx))
    wf[1, 2*nsamples-i+1] = data/10*sqrtx * math.cos(turns/5*math.pi*(100 + sqrtx))
    
  return Waveform(wf, step)

#def helixscan2(data, turns):
  #"""Returns a closed helix waveform. """
  #data=float(data)
  #turns=float(turns)
  #step = 100/(math.pi * turns * turns)
  #nsamples = int(100 / step)
  #wf = np.empty(shape=(2, 2*nsamples+2), dtype=float)
  #def r(t):
    #return t
  #def theta(t):
    #return math.log(t+1)
  #for i in xrange(nsamples+1):
    #wf[0, i] = r(t) * math.sin(2*math.pi*theta(t))
    #wf[1, i] = r(t) * math.cos(2*math.pi*theta(t))
    #wf[0, 2*nsamples-i+1] = data/10*sqrtx * math.sin(turns/5*math.pi*(100 - sqrtx))
    #wf[1, 2*nsamples-i+1] = data/10*sqrtx * math.cos(turns/5*math.pi*(100 + sqrtx))
    
  #return Waveform(wf, step)

def gosine(a, b, dt, rate, t0=0):
  """From A to B in a graceful fashion.
  
  This is done with a cosine arc :math:`y(t) = a + (b-a) (1-\cos(2 \pi f t))/2`
  
  :param a: starting value
  :param b: final value
  :param dt: time interval between a and b
  :param rate: waveform sampling rate
  :param t0: starting time at a
  :rtype: :class:`Waveform`
  
  """
  (a, b, dt, rate, t0) = map(float, (a, b, dt, rate, t0))
  f = 1/(2*dt)
  time = np.arange(t0, t0+dt, 1/rate, dtype=float)
  out = np.empty(shape=time.size, dtype=float)
  for i, t in enumerate(time):
    out[i] = a + (b-a)*(1-math.cos(twopi*f*t))/2
  return Waveform(out, 1/rate)

def linscan(amplitude=1, lines=100, fscan=10, rate=10000, cap=0.08):
  """Return a two-channel quasi-sinusoidal waveform made of linear segments joined by sine curves.
  
  :param amplitude: amplitude of the waveform
  :param lines: number of periods
  :param fscan: frequency of the waveform
  :param rate: sampling rate
  :param cap: fraction of the waveform spend in the sine cap.
              between 0 (triangular wave) and 1 (pure sine wave)
  :rtype: :class:`Waveform`
  
  The :attr:`Waveform.linear_subset` attribute of the returned object is set
  to indicate the linear portions of the waveform (this is useful when using
  this waveform to drive a :class:`usb2600.usb2600.USB2600_Sync_AIO_Scan` scan).
  
  :Exemple:
  
  >>> a = waveform.linscan(1,10,10,1000,0.08)
  >>> a.plot()
  >>> a.plotxy()
  
  produces the two-channel waveform shown below. The right figure shows
  the corresponding 2D scan pattern when the two channel are driving
  X and Y scanning mirrors (for example).
  
  .. image:: /../mccdaq/utilities/img/linscan_demo_X,Y.png
     :height: 250px
  .. image:: /../mccdaq/utilities/img/linscan_demo_XY.png
     :height: 250px
  """
  rate = float(rate)
  fscan = float(fscan)
  amplitude = float(amplitude)
  n=lines
  fcap = fscan * (4*cap + 2/math.tan(twopi*cap)/math.pi)
  tcap = cap/fcap
  tlin = 1/(math.tan(twopi*cap) * fcap * math.pi)
  per = 1/fscan
  time = np.arange(0, (n+1)*per, 1/rate, dtype=float)
  print (time.size)
  out = np.empty(shape=(2, time.size), dtype=float)
  #per = 4*tcap + 2*tlin
  print (rate, fscan, amplitude, fcap, tcap, tlin, per)
  lin = []  # list of indices indicating the linear portions of the waveform.
  in_lin = False
  
  start = np.vstack((gosine(0., 1., per/2., rate).data, 
                     gosine(0., -1., per/2., rate).data))
  end = np.vstack((gosine(1., 0., per/2., rate).data, 
                   gosine(1., 0., per/2., rate).data))
  lstart = start.size/2
  
  for i in xrange(time.size):

    (m, t) = divmod(time[i]+(0.01/rate), per)
    a = max(-1., (m-1)*2/n - 1)
    b = min(1., m*2/n -1)

    if t <= tcap:
      if in_lin:
        lin.append(i+lstart)
        in_lin = not in_lin
      out[0,i] = math.cos(twopi*t*fcap)
      out[1,i] = a
    elif tcap < t <= tcap + tlin:
      if not in_lin: 
        lin.append(i+lstart)
        in_lin = not in_lin
      out[0,i] = -twopi*fcap*math.sin(twopi*cap)*(t-tcap)+math.cos(twopi*cap)
      out[1,i] = a
    elif tcap + tlin < t <= 3*tcap + tlin:
      if in_lin:
        lin.append(i+lstart)
        in_lin = not in_lin
      out[0, i] = -math.cos(twopi*(t-(2*tcap+tlin))*fcap)
      out[1,i] = a + 0.5*(b-a)*(1-math.cos(twopi/(4*tcap)*(t-(tcap+tlin))))
    elif 3*tcap + tlin < t <= 3*tcap + 2*tlin:
      if not in_lin:
        lin.append(i+lstart)
        in_lin = not in_lin
      out[0,i] = twopi*fcap*math.sin(twopi*cap)*(t-(3*tcap+2*tlin))+math.cos(twopi*cap)
      out[1,i] = b
    elif 3*tcap + 2*tlin < t < 4*tcap + 2*tlin:
      if in_lin:
        lin.append(i+lstart)
        in_lin = not in_lin
      out[0,i] = math.cos(twopi*(t-(4*tcap+2*tlin))*fcap)
      out[1,i] = b
      
  wf = Waveform(amplitude * np.hstack((start, out, end)), 1/rate)
  wf.linear_subset = lin
  return wf
