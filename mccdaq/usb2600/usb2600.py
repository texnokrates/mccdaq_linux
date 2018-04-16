"""
High-level, object-oriented Python interface for MCCDAQ's USB2600 data acquisition devices.

Built on top of the low-level Cython :mod:`usb2600.wrapper` module.

It includes advanced functionalities for analog input and output scans
of arbitrary length and synchronous analog input/output scanning.

.. todo::

   Add support for ``retrigger`` and ``burst`` options

:Examples:

>>> from mccdaq import usb2600
>>> daq = usb2600.USB2600()   # Initialise device
>>> daq.device.blink(10)      # blink LED 10 times
>>> daq.device.temperature()  # read temperature
>>> daq.status                # return info about AI/AO scans

:Counter:

>>> daq.counter0.init()  # Initialiase Counter 0
>>> daq.counter0.read()  # Read number of detected edges since init()

:Timer:

>>> daq.timer0.configure(frequency=1000, duty_cycle=0.25) #1 kHz, 0.25 duty cycle square waveform
>>> daq.timer0.start()
>>> daq.timer0.stop()

:Analog input:

>>> v = daq.ai0.read()         # Single sample from AI0
>>> v = daq.ai.read((1,4,16))  # Single sample from multiple channels
>>> aiscan = daq.AIScan((0,1,2), nsamples=1000, rate=1000)  # AI scan: 1kS at 1kS/s from channels 0,1 and 2
>>> aiscan.run()           # Run the scan
>>> aiscan.data            # Acquired data (numpy array)

:Analog output:

>>> daq.ao0(1)           # Set AO0 to 1V
>>> wf = waveform.sine(10, nsamples=1000, rate=1000) # Sine waveform, 10Hz, 1kS @ 1kS/s
>>> aoscan = daq.AOScan((3,), wf)  # Set up analog output of waveforms in data
>>> aoscan.run(thread=True)    # run the scan in background
>>> aoscan.run(n=10)           # output the waveform 10 times
>>> aoscan.run(n=0)           # output the waveform forever

:Synchronous analog input/output:

>>> wf = waveform.helixscan(1, 10)         # Create 2D waveform
>>> aio = daq.AIOScan((0,1,2), (0,1), wf)  # Output waveform on AO channels 0 and 1 while acquiring on AI channels 0, 1 and 2
>>> aio.run()                              # start scan
>>> aio.data.plot()                        # visualise acquired data

:Triggering analog scans:

>>> daq.external_trigger.configure()           # Configure external trigger
>>> daq.internal_trigger.configure(daq.dio.A7) # Configure internal trigger, source is A7

:Digital input/outputs:

>>> daq.dio.A.direction = 0x0F  # Configure bits 7-4 as outputs, 3-0 as input on port A
>>> daq.dio.A()                 # Read whole port
>>> daq.dio.A(0xAA)             # Set all bits on port
>>> daq.dio.A0.isoutput = True  # Configure pin A0 as output
>>> daq.dio.A1.isinput = True   # Configure pin A1 as input
>>> # Several methods to set an output pin
>>> daq.dio.A0(True)          # pin is callable
>>> daq.dio.A0.set()          # set() = HIGH
>>> daq.dio.A0.reset()        # reset() = LOW
>>> daq.dio.A0.state = True   # property
>>> # Several methods to read an input pin
>>> daq.dio.A1()              # pin is callable
>>> daq.dio.A1.get()          # method
>>> daq.dio.A1.state          # property


.. note::
   If the :mod:`waveform` module is not available, :class:`AO_Scan` and 
   :class:`AIO_Scan` will not work.

(c) 2014 Guillaume Lepert, Imperial College London.

----------------------------------
"""

# v1.1 : Create a package
# v1.2 : Fix Trigger class to support internal triggering.
#        Rewrite AO, AI and AIO Scan accordingly
#        Add waveform module to package
# v1.3 : Add _run_all_samples() method to AOScan
# v1.4 : Improve triggering
#        Add individual DIO pins access

# Known bugs:
#   - AI scan will fail if number of samples is less than packet_size

from wrapper import USB2600_Wrapper

import numpy as np

import threading
from time import sleep, clock
import math
import sys
import warnings
from ..utilities import waveform
print waveform
#try:
#  from ..utilities import waveform
#  #import waveform
#  _WITH_WAVEFORM = True
#except ImportError:
#  warnings.warn("Module 'waveform' not found. Analog input/output scan will not available.")
#  _WITH_WAVEFORM = False


class USB2600Error(RuntimeError):
  """Alias of :exc:`exceptions.RuntimeError` to represent USB2600 exceptions."""
  pass

class USB2600(object):
  """High-level interface for MCCDAQ USB-2600 series data acquisition device.
  
  This top-level container provides access to:
    - analog inputs ``ai0``, ``ai1``, ... are instances of :class:`AI_Single`.
    - ``ai`` represents several analog input channels to be read simultaneously: see :class:`AI_Multichannel`.
    - analog outputs ``ao0``, ``ao1``, ... are instances of :class:`AO_Single`.
    - digital input/output ports ``dio.A``, ``dio.B``, ..., are instances of :class:`DIO_Port`.
      Each contains 8 ``pin0``, ``pin1``, etc. , instances of :class:`DIO_Pin`. 
    - counters ``counter0``, ``counter1``, ... are instances of :class:`Counter`.
    - timers ``timer0``, ``timer1``, ... are instances of :class:`Timer`.
    
  Create analog scans (AI, AO and synchronous AIO) with the methods
  :func:`AIScan`, :func:`AOScan` and :func:`AIOScan`.
    
  Blink the LED with :func:`blink`.
  
  Read the device :attr:`temperature`, :attr:`serial_number`, and :attr:`status`.  
  """
  def __init__(self, device=None, model=2627):
    """
    :param device: a :class:`usb2600.wrapper.USB2600_Wrapper` instance. If None, creates a new instance.
    :param int model: the model of the device to look for (2623, 2627, 2633 or 2637)
    """
    if device is None:  
      self.device = USB2600_Wrapper(model)     # Instantiate device
    else:
      self.device = device
    
    # Instantiate the counters
    for i in range(4):
      setattr(self, 'counter'+str(i), Counter(self.device, i))
    
    # Instantiate the timers 
    for i in range(4):
      setattr(self, 'timer'+str(i), Timer(self.device, i))
    
    # Instantiate the 8-bits digital input/output ports
    #for i, p in enumerate(('A', 'B', 'C')):
    #  setattr(self, 'dio'+p, DIO_Port(self.device, i))
  
    self.dio = DIO(self.device)
    
    # Instantiate the Analog inputs (single)
    for i in range(16):
      setattr(self, 'ai'+str(i), AI_Single(self.device, i))
    # Instantiate the multichannel analog input object
    self.ai = AI_Multichannel(self.device)

    # Instantiate the Analog outputs 
    for i in range(4):
      setattr(self, 'ao'+str(i), AO_Single(self.device, i))

    # Default internal trigger
    self.internal_trigger = InternalTrigger(self.dio.A7)

    # Default external trigger
    self.external_trigger = ExternalTrigger(self.device)

  def __repr__(self):
    return "<MCC USB-2627, S/N: " + self.serial_number + " (object-oriented interface)>"
  
  @property
  def serial_number(self):
    """Return the device's serial number."""
    return self.device.serialNumber()

  @property
  def temperature(self):
    """Return the device internal temperature, in Celsius."""
    return self.device.temperature()

  def blink(self, n=1):
    """Blink the LED *n* times."""
    self.device.blink(n)
  
  @property
  def status(self):
    """Return the device status as a dictionary."""
    return self.device.status()
    
  def _check_FIFO_status(self, status=None):
    """Raise a USB2600 Exception if AI/AO FIFO has over/underrun.
    
    :param status: the status dict returned by status().
                   If None, call :func:`status` to get it.
    :raise: :class:`USB2600Error`
    
    Note: not currently in use.
    """
    if status is None:
      status = self.status
    if status['AIN_SCAN_OVERRUN']:
      raise USB2600Error('Analog input FIFO overrun! Consider decreasing tf parameter.')
    if status['AOUT_SCAN_UNDERRUN']:
      raise USB2600Error('Analog output FIFO underrun! Consider decreasing tf parameter.')
  
  def AIScan(self, channels, nsamples, rate, trigger=False, retrigger=1,
             burst = True, packet_size = 240, tf = 0.5, verbose=False):
    """Returns an Analog Input object (:class:`AI_Scan`)
    
    :param channels: tuple of integers indicatings the analog input channels.
    :param nsamples: number of samples to acquire
    :param rate: sampling frequency (Hz)
    :param trigger: if True, start the scan on XTTLTRG.
    """
    return AI_Scan(self.device, channels, nsamples, rate,
                          trigger, retrigger,
                          burst, packet_size, tf, verbose)
  
  def AOScan(self, channels, waveform, rate=None, trigger=False, retrigger=1,
               packet_size = 240, tf = 0.5, verbose=False):
    """ Returns an Analog Output object (see :class:`AO_Scan`)
    
    :param channels: tuple of integers indicatings the analog output channels.
    :param waveform: a :class:`Waveform` instance containing the data.
                         Must have the same number of rows as *channels*
    :param rate: sampling frequency (Hz). if not given, use that of the waveform.
    :param trigger: if True, start the scan on XTTLTRG.
    """
    return AO_Scan(self.device, channels, waveform, rate, trigger, retrigger,
                           packet_size, tf, verbose)
  
  def AIOScan(self, AIchans, AOchans, AOwaveform, rate=None,
                         packet_size = 240, tf = 0.5, verbose=False):
    """Return a Synchronous Analog Input/Output scan object (:class:`AIO_Scan`).
    
    :param AIchans: tuple of integers indicatings the analog input channels.
    :param AOchans: tuple of integers indicatings the analog ouput channels.
    :param AOdata: :class:`Waveform` object
               
    .. Note::    
       - The sampling frequency is taken from the ``AOdata`` waveform.
       - The trigger's digital output must be connected to the external trigger input XTTLTRG.
    """
    return AIO_Scan(self, AIchans, AOchans, AOwaveform, rate,
                                 packet_size, tf, verbose)

 
class Timer(object):
  """Each timer on the device is an instance of this class."""
  def __init__(self, device, timer):
    """
    :param device: the :class:`USB2600` instance
    :param timer: timer index on the device
    :type timer: int
    """
    self._device = device
    self.timer = timer
    self.is_configured = False
    self.is_running = False
  
  def __repr__(self):
    if self.is_configured:
      return "USB2600 Timer %(timer)d configuration: %(frequency)0.1f Hz, duty cycle %(duty_cycle)0.2f, %(counts)s pulses." % self.__dict__
    else:
      return "USB2600 Timer %(timer), unconfigured." % self.__dict__
    
  def status(self, read_configuration = False):
    """Check the timer status (Enable, Running).
    
    Optionally query the current configuration (Period, Pulse width, Delay, Count)
    and writes to the corresponding attributes.
    """
    if read_configuration == True:
      self.period = self._device.timerPeriodRead(self.timer)
      pw = self._device.timerPulseWidthRead(self.timer)
      self.duty = pw / float(self.period)
      self.delay =  self._device.timerDelayRead(self.timer)
      self.counts = self._device.timerCountRead(self.timer)
    return self._device.timerControlRead(self.timer)
  
  def configure(self, frequency=None, period=None, duty_cycle=0.5, pulse_width=None, counts=0, delay=0):
    """Configure the timer (either frequency (in Hz) or period (in units of the 64MHz clock cycles) can be specified.
    
    :param float frequency: in Hz
    :param int period: in units of the 64MHz clock
    :param float duty_cycle: [0...1] (default 0.5)
    :param int pulse_width: in units of the 64MHz clock
    :param int count: number of pulses to output (0=forever, default)
    :param int delay: delay before starting timer, in units of clock cycles.
    """
    assert (frequency is not None) != (period is not None), "Either frequency or period must be specified."
    if self.is_running:
      raise Exception("Cannot reconfigure running timer! Stop it first.")
    
    # Calculate frequency/period
    if period is None:
      self.period = int(64.0E6/frequency - 1)
      self.frequency = frequency
    else:
      self.frequency = 64.0E6/(float(period+1))
      self.period = period
    
    # Calculate duty cycle/pulse width
    if pulse_width is None:
      self.pulse_width = int(period * duty_cycle)
      self.duty_cycle = duty_cycle
    else:
      self.duty_cycle = float(pulse_width) / float(self.period)
      self.pulse_width = pulse_width

    self._device.timerPeriodWrite(self.timer, self.period)
    self._device.timerPulseWidthWrite(self.timer, self.pulse_width)
    self._device.timerDelayWrite(self.timer, delay)
    self.delay = delay
    self._device.timerCountWrite(self.timer, counts)
    self.counts = counts
    self.is_configured = True
    
  def start(self, invert=False):
    """Start the timer."""
    if not self.is_configured:
      raise RuntimeError("Cannot start timer: not configured")
    if self.is_running:
      raise RuntimeError("Cannot start timer: already running!")
    self._device.timerControlWrite(self.timer, enable=True, invert=invert)
    self.is_running = True
  
  def stop(self):
    """Stop the timer."""
    self._device.timerControlWrite(self.timer, enable=False)
    self.is_running = False


class Counter(object):
  """ Each counter on the device will be an instance of this class. """
  
  def __init__(self, device, counter):
    """
    :param device: the :class:`USB2600` instance
    :param counter: counter index on the device
    :type counter: int
    """
    self._counter = counter
    self._device = device
    
  def init(self):
    """Initialise the counter."""
    self._device.counterInit(self._counter)
    
  def read(self):
    """Return the number of counts since the counter was initialised."""
    return self._device.counterRead(self._counter)


class DIO(object):
  """Container for DIO ports and pins."""
  def __init__(self, device):
    for (port_index, port_name) in enumerate('ABC'):
      _port = DIO_Port(device, port_index)
      setattr(self, port_name, DIO_Port(device, port_index))
      for pin in range(8):
        setattr(self, port_name+str(pin), DIO_Pin(_port, pin))

  def __repr__(self):
    return "<USB2600 DIO container>"


class DIO_Port(object):
  """Controls a DIO port."""
  def __init__(self, device, port):
    """
    :param device: the :class:`USB2600` instance
    :param int port: port index on the device
    """
    self._device = device
    self._port = port
    for i in range(8):
      setattr(self, 'pin'+str(i), DIO_Pin(self, i))
  
  def __repr__(self):
    return "<USB2600 DIO_Port %s>" % "ABC"[self._port]
  
  @property
  def state(self):
    """Set or query the port state."""
    return self._device.dioRead(self._port)
  @state.setter
  def state(self, byte):
    self._device.dioWrite(self._port, byte)

  @property
  def direction(self):
    """Query or configure the port direction (0xFF all inputs, 0x00 all ouputs, or a mixture: 0x01, etc.)"""
    return self._device.dioConfig(self._port)
  @direction.setter
  def direction(self, byte):      
    self._device.dioConfig(self._port, byte)
    
  def read(self):
    """Read the port state."""
    return self.state
      
  def write(self, byte):
    """Write *byte* to the port."""
    self.state = byte
  
  def __call__(self, byte=None):
    """Set or query the port state."""
    if byte is None:
      return self.state
    else:
      self.state = byte

def set_bit_in_byte(byte, bit, state):
  return byte & ~(1 << bit) | (state << bit)

def get_bit_in_byte(byte, bit):
  return bool((byte >> bit) & 1)


class DIO_Pin(object):
  """Controls a single bit of a DIO port."""
  def __init__(self, port, pin):
    """
    :param port: a :class:`DIO_Port` instance
    :param int pin: the target pin on ``port``
    """
    self._device = port._device
    self._port = port
    self._port_index = port._port
    self._pin = pin
    
  def __repr__(self):
    return "<USB2600 DIO_Pin %s%d>" % ("ABC"[self._port_index], self._pin)

  def get(self):
    """Return the pin state."""
    return self.state
  
  def set(self):
    """Set the pin to High."""
    self.state = True

  def reset(self):
    """Reset the pin to Low."""
    self.state = False
  
  def __call__(self, boolean=None):
    """Set or query the pin state."""
    if boolean is None:
      return self.state
    else:
      self.state = boolean
      
  @property
  def state(self):
    """Set or query the pin state."""
    return bool((self._port.state >> self._pin) & 1)
  @state.setter
  def state(self, boolean):
    current = self._port.state
    self._port.state = (current & ~(1 << self._pin) | (boolean << self._pin))

  @property
  def isinput(self):
    """Query whether the pin is configured as input, or configure it so by setting it to True."""
    return bool((self._port.direction >> self._pin) & 1)
  @isinput.setter
  def isinput(self, boolean):
    current = self._port.direction
    byte_out = current & ~(1 << self._pin) | (boolean << self._pin)
    self._port.direction = byte_out

  @property
  def isoutput(self):
    """Query whether the pin is configured as output, or configure it so by setting it to True."""
    return not self.isinput
  @isoutput.setter
  def isoutput(self, boolean):
    self.isinput = not boolean


class ExternalTrigger(object):
  """Represents an external trigger (external digital output connected to XTTLTRG)."""
  def __init__(self, device, mode='Edge', polarity='Rising'):
    """
    :param mode: 'Edge' or 'Level'.
    :param polarity: in 'Edge' mode:  'Rising' or 'Falling'.
                     In 'Level' mode : 'Low' or 'High'.
    """
    self._device = device
    self.mode_string = mode
    self.polarity_string = polarity
    # No need to call configure(): device defaults as indicated.

  def __repr__(self):
    return "USB2600 external trigger: " + self.polarity_string + ' ' + self.mode_string

  def configure(self, mode='Edge', polarity='Rising'):
    """
    :param mode: 'Edge' or 'Level'.
    :param polarity: in 'Edge' mode:  'Rising' or 'Falling'.
                     In 'Level' mode : 'Low' or 'High'.
    """
    if mode == 'Edge':
      self.mode = 1
    elif mode == 'Level':
      self.mode = 0
    else:
      raise ValueError('Trigger mode must be "Edge" or "Level".')
    if polarity in ('Rising', 'High'): 
      self.polarity = 1
    elif polarity in ('Falling', 'Low'):
      self.polarity = 0
    else:
      raise ValueError('Trigger polarity should be "Rising" or "Falling", or "High" or "Low".')
    self.mode_string = mode
    self.polarity_string = polarity
    self._device.TriggerConfig(mode=self.mode_string, polarity=self.polarity_string)


class InternalTrigger(object):
  """Internal trigger by connection XTTLTRG to a digital output pin."""
  def __init__(self, dio_pin, mode='Edge', polarity='Rising'):
    """
    :param dio_pin: a :class:`DIO_Pin` instance, physically connected to XTTLTRG.
    :param mode: 'Edge' or 'Level'.
    :param polarity: in 'Edge' mode:  'Rising' or 'Falling'.
                     In 'Level' mode : 'Low' or 'High'.
    """
    self._device = dio_pin._device
    self.trigger_input = ExternalTrigger(self._device, mode, polarity)
    self.configure(dio_pin, mode, polarity)
    
  def __repr__(self):
    return "USB2600 internal trigger: " + self.polarity_string + ' ' + self.mode_string

  def configure(self, dio_pin, mode='Edge', polarity='Rising'):
    self.trigger_ouput = dio_pin
    self.trigger_ouput.isoutput = True
    self.trigger_input.configure(mode, polarity)
    self.reset()

  def trigger(self):
    """Fire the trigger."""
    self.trigger_ouput.state = self.trigger_input.polarity
    
  def reset(self):
    """Reset the trigger so it can be fired again."""
    self.trigger_ouput.state = not self.trigger_input.polarity


class AI_Single(object):
  """Single sample, single channel analog input.
  
  >>> volts = daq.ai0.read()
  >>> volts = daq.ai0()      # callable.
  """
  def __init__(self, device, channel):
    self._device = device
    self._channel = channel
  
  def read(self, nsamples=1, dt=1):
    """Read analog input samples.
    
    :param nsamples: number of samples to acquire
    :param dt: time (in s, default 1) between samples.
    """
    # Channel is configured in low-level AIn() function
    self.last_reading = self._device.AIn(self._channel)
    if nsamples > 1:
      samples = [self.last_reading]
      for i in range(nsamples):
        sleep(dt)
        self.last_reading = self._device.AIn(self._channel)
        samples.append(self.last_reading)
      return samples
    else:  
      return self.last_reading
  
  def __call__(self):
    """Read single analog input sample."""
    return self.read()


class AI_Multichannel(object):
  """ Single sample, multiple channel acquisition: out = daq.ai.read((ch1, ..., chn)). """

  def __init__(self, device):
    self._device = device
  
  def read(self, channels):
    """ Single sample, single channel analog input. """
    # Channel is configured in low-level AIn() function
    readings = []
    for ch in channels:
      readings.append(self._device.AIn(ch))
    return readings
  
  # try not calling reconfigure every time???
  def read_fast(self, channels):
    #if self._device.last_ai_config != channels:
    self._device.AInConfig(channels)
    readings = []
    for ch in channels:
      readings.append(self._device.AIn(ch))
    return readings
  
class AI_Scan(object):
  """Analog Input scan.

  Acquired data in :attr:`data`.

  TO DO: return data as a Waveform object."""
  def __init__(self, device, channels, nsamples, rate, trigger=False, retrigger=1,
               burst=True, packet_size=240, tf=0.5, verbose=True):
    """
    :param device: :class:`USB2600` instance
    :param channels: tuple of integers indicatings the analog input channels.
    :param nsamples: number of samples to acquire
    :param rate: sampling frequency (Hz). To use an external clock source
                 (connected to ``AI_CLK_IN`` on the board), set ``rate=0``.
    :param trigger: Whether to wait for a trigger on XTTLTRG to start acquiring.
                    If False, acquisition starts when calling :func:`run`
    :param retrigger: How often
    :param burst: burst mode (**UNTESTED**)
    :param packet_size: number of bytes in each USB read operation.
    :param tf: USB read time factor. If getting ``AI_FIFO_OVERRUN``, try 
               decreasing *tf* to pop data from the FIFO faster.
    :param verbose: if True, print info to console while scan is running.
    """
    self._device = device
    self.channels = channels
    self.nchannels = len(channels)

    self.rate = rate
    self.trigger = trigger
    self.retrigger = retrigger
    self.burst = burst
    self.packetSize = packet_size / self.nchannels  # Split the request packet size between all channels
    # nsamples must be a multiple of packet_size
    if nsamples % self.packetSize > 0:
      self.nsamples = nsamples + self.packetSize - (nsamples % self.packetSize)
    else:
      self.nsamples = nsamples
    self.tf = tf
    self.last_sample_read = 0
    self.all_samples_read = False
    self.throughput = self.nchannels * float(self.rate)
    self.t_update = self.nchannels * self.packetSize / self.throughput
    self.n_updates = self.nsamples / self.packetSize
    self.verbose = verbose
    self.__call__ = self.run
  
  def __repr__(self):
    return "<USB2600 AI scan instance on channels %(channels)s: %(nsamples)d samples at %(rate)0.2f Hz.>" % self.__dict__ 
            
  def config(self):
    """Configure the AI scan. """
    self._device.AInConfig(self.channels)
    
  def start(self):
    """ Start the AI scan. """
    self.data = np.empty(shape=(self.nchannels, self.nsamples), dtype=np.float)
    self._device.AInScanStart(self.channels, self.nsamples, self.rate,
                 burst=self.burst, trigger=self.trigger, retrigger=self.retrigger, 
                 packet_size=self.nchannels*self.packetSize-1)
    if self.verbose: print "Started the AI scan."
    # reinitialise counters
    self.last_sample_read = 0
    self.all_samples_read = False
    self.stopped = False
  
  def read_basic(self, nsamples):
    """ Read nsamples per channel from FIFO. """
    return self._device.AInScanRead(self.nchannels, nsamples)
  
  def read(self):
    """ Read a single packet of data from FIFO. """
    start = self.last_sample_read
    end = min(start + self.packetSize, self.nsamples)
    (data_part, bytes_read) =  self._device.AInScanRead(self.nchannels, min(self.packetSize, self.nsamples - self.last_sample_read))
    self.data[:, start:end] = data_part
    self.last_sample_read = end   #  Update counter
    if bytes_read != 2 * self.nchannels * (end - start):
      raise RuntimeError('Did not get all expected data!')
    if self.verbose:
      print('Got ' + str(data_part.size) + ' samples (' + str(bytes_read) + ' bytes) from AI FIFO.')
    if end == self.nsamples:
      self.all_samples_read = True
      if self.verbose: print("All AI samples have been read.")
    return bytes_read/2 # return no of samples read
  
  def stop(self):
    """Stop the AI scan. """ 
    self._device.AInScanStop()
    self.stopped = True
    
  def clear(self):
    """Clear the AI FIFO. Data will be lost! """
    self._device.AInScanClear()
    
  def flush(self):
    """Read all remaining elements in FIFO.
    
    It may be necessary to call flush() before an AI or AIO scan
    to remove old, unread samples from AI FIFO.
    """
    b = 0
    while True:
      # read packets from FIFO until it's empty and times out.
      try:
        (data, bytes_read) = self._device.AInScanRead(self.nchannels, self.packetSize)
        b += bytes_read
        if self.verbose:
          sys.stdout.write('Purging AI FIFO... %d bytes read\r' %bytes_read)
          sys.stdout.flush()
      except:
        if self.verbose: print("AI FIFO flushed, "+str(b)+" bytes read.                        ")
        break
    
  def print_settings(self):
    print('AI throughput = '+ str(self.throughput) + ' S/s. '+ 
          'Read ' + str(self.packetSize * self.nchannels) + ' samples from FIFO every ' + str(self.t_update) + ' s' +
          ' in ' + str(self.n_updates) + ' iterations.')
  
  def print_status(self):
    if self.running:
      self.ai_status = 'AI Scan running, %(last_sample_read)d/%(nsamples)d read. [Ctrl-C to stop]' % self.__dict__
    elif self.done:
      self.ai_status = 'AI Scan done, %(last_sample_read)d/%(nsamples)d samples read.' % self.__dict__
    else:
      self.ai_status = "AI neither running nor done, I'm losing it!!!"
    if (not self.threaded): # and self.verbose:
      sys.stdout.write(self.ai_status + '\r')
      sys.stdout.flush()

  def _run(self):
    """Run a complete AI acquisition."""    
    self.stop()
    self.clear()
    self.config()
    self.last_sample_read = 0
    self.all_samples_read = False
    self.print_settings()
    self.start()
    sleep(3*self.t_update)
    
    try:
      while (not self.stopped) and (not self.all_samples_read):
        sleep(self.tf * self.t_update)
        self.read()
        self.print_status()
        if self.fifo_overrun:
          raise USB2600Error('Analog input FIFO overrun! ')
    except KeyboardInterrupt:
      print "\nScan interrupted by user."
      self.stop()
      self.clear()
    global _WITH_WAVEFORM
    if _WITH_WAVEFORM:
      self.data = waveform.Waveform(self.data, self.rate)
      
  def run(self, thread=False):
    """Start the scan in main or background thread.
    
    In a background thread, call stop() to stop the scan before completion.
    In the main thread, use Ctl+C."""
    if thread:
      self.threaded = True
      self.verbose = False
      self.thread = threading.Thread(target=self._run, args=(), name='USB2600_AI_scan')
      self.thread.start()
    else:
      self.threaded = False
      self._run()

  @property
  def fifo_overrun(self):
    """Return True if the AI FIFO is full."""
    return self._device.status()['AIN_SCAN_OVERRUN'] == 1
    
  @property
  def running(self):
    """Return True while samples are being read."""
    return self._device.status()['AIN_SCAN_RUNNING'] == 1

  @property
  def done(self):
    """Return True once all samples have been read."""
    return self._device.status()['AIN_SCAN_DONE'] == 1


class AO_Single(object):
  """Single sample, single channel analog ouput.
  
  >>> daq.ao0.write(0.56) # output 0.56 volts to AO0
  """
  def __init__(self, device, channel):
    self._device = device
    self._channel = channel
    self.limits = (-10, +10)
    
  def write(self, voltage):
    """Set the AO voltage. """
    assert self.limits[0] <= voltage <= self.limits[1], 'Requested voltage %r V ouside allowed limits %s' %(voltage, str(self.limits))
    self.last_output = self._device.AOut(self._channel, voltage)
  
  def read(self):
    """Read the curreny AO voltage."""
    return self._device.AOutRead(self._channel)
   
  def __call__(self, voltage=None):
    """Set or query (if no argument is supplied) the output voltage."""
    if voltage is None:
      return self.read()
    else:
      self.write(voltage)


class AO_Scan(object):
  """Single- or multi-channel analog output scan.
  
  To start the scan:
  
  >>> run(n=0, thread=True|False)
    * n: output the waveform n times (0=infinity)
    * thread: whether to run in a background thread (call stop() to stop it)
      or in the main thread (Ctl+C to stop)
  """
  
  def __init__(self, device, channels, waveform, rate=None, trigger=False, retrigger=1,
               packet_size=240, tf=0.5, verbose=False):
    """
    :param device: :class:`USB2600` instance
    :param channels: tuple of integers indicatings the analog output channels.
    :param waveform: a :class:`Waveform` instance containing the data.
                     Must have the same number of rows as ``channels``
    :param rate: sampling frequency (Hz). if not given, use ``waveform.data``.
    :param trigger: a :class:`Trigger` instance to use as external trigger.
    :param packet_size: number of bytes in each USB read operation.
    :param tf: USB read time factor. If getting ``AO_FIFO_UNDERRUN`` error, 
               try decreasing *tf* to push data to the FIFO faster.
    :param verbose: if True, print info to console while scan is running."""
    self._device = device
    self.channels = channels
    self.waveform = waveform
    assert waveform.nchannels == len(channels), 'Number of waveforms must match number of AO channels'
    self.nchannels = len(channels)
    self.nsamples = waveform.nsamples
    self.total_samples = self.nsamples*self.nchannels
    self.rate = waveform.rate if rate is None else float(rate)
    self.data = self.get_waveform_data(waveform)
    self.trigger = trigger
    self.retrigger = retrigger
    self.packetSize = packet_size
    self.tf = tf
    self.last_sample_written = 0
    self.timeout = 1000
    self.throughput = self.nchannels * self.rate
    self.n_updates = self.data.size / self.packetSize
    self.t_update = self.packetSize / self.throughput
    self.all_samples_written = False
    self.verbose = verbose
    self.__call__ = self.run
    
  def get_waveform_data(self, waveform):
    """Return a flattened and binary version of the waveform data."""
    if waveform.data.ndim == 2:
      data = np.reshape(waveform.data, waveform.data.size, 'F')
    else:
      data = waveform.data
    if data.dtype == np.float:
      data = self._device.AOutData(data)
    return data
    
  def __repr__(self):
    return "<USB2600 AO scan instance on channels %(channels)s: %(nsamples)d samples at %(rate)0.2f Hz.>" % self.__dict__ 
  
  def start(self, repeat=1, continuous=False):
    """Start the AO scan.
    
    :param continuous: True for continuous operation (output until FIFO underruns).
                       False to output a fixed number of samples, determined by the data waveform
                       and the repeat parameter.
    :param repeat: number of time to output the waveform. 0 if forever.
    """
    nsamples = 0 if continuous else self.nsamples*repeat
    self._device.AOutScanStart(self.channels, self.rate, nsamples,
                 trigger=self.trigger, retrigger=self.retrigger)
    if self.verbose:
      print('Started the AO scan.')
    self.stopped = False

  def write(self, n = 1):
    """Write *n* packet of samples to the AO FIFO."""
    start = self.last_sample_written
    end = min(start + n * self.packetSize, self.data.size)
    bytes_written = self._device.AOutDataToFIFO(self.data[start:end], timeout=self.timeout)
    if bytes_written/2 != end - start:
      raise USB2600Error(str(end - start) +' samples send, only '+str(bytes_written/2) + ' written.')
    self.last_sample_written = end
    if self.verbose:
      print(str(bytes_written) + ' bytes written to AO FIFO')
    if end == self.data.size:
      self.all_samples_written = True
      if self.verbose:
        print("All AO samples have been written")
    return bytes_written/2
  
  def stop(self):
    """Stop the AO scan.""" 
    self._device.AOutScanStop()
    self.stopped = True
    
  def clear(self):
    """ Clear the AO FIFO. Data will be lost! """
    self._device.AOutScanClear()
    
  def print_settings(self):
    print("Write %(packetSize)d samples to FIFO every %(t_update)ds. \
           %(n_updates)d iterations/%(nsamples)d samples in total." % self.__dict__)
  
  def print_status(self):
    """Pretty-print scan status on a single line."""
    if self.running:
      self.string = "AO: running, %(last_sample_written)d/%(total_samples)d samples written to FIFO. [Ctrl-C to stop]" %self.__dict__
    else: 
      self.string = "AO: Done."
    sys.stdout.write(self.string + ("\n" if self.verbose else "\r"))
    sys.stdout.flush()

  def write_waveform_to_fifo(self, n=1):
    """Write the whole waveform to FIFO n times (forever if n=0)."""
    counter = 1 # increase by 1 everytime a full waveform has been written to FIFO.
    
    while (not self.stopped) and (counter <= n or n==0):
      if self.fifo_underrun:
        raise USB2600Error('Analog output FIFO underrun! Maybe decrease tf=%r parameter?' %self.tf)
      sleep(self.tf * self.t_update)
      if not self.all_samples_written:
        self.write()  # push one packet and update counter
      if not self.threaded:
        self.print_status()
      if self.all_samples_written:  # reset counter after all samples have been written
        self.all_samples_written = False
        self.last_sample_written = 0
        counter += 1
        
  def wait_until_output_done(self):
    """Return after all samples have been written to output."""
    while self.running:
      sleep(self.tf * self.t_update)
      if not self.threaded:
        self.print_status()
  
  def _run_split_samples(self, n=1, wait=False):
    """Outputs the waveforms n times (forever if n=0).
    
    Samples will be pushed to AO FIFO in packet_size groups.
    Use :func:`_run_all_samples` instead when all samples can fit in FIFO."""
    if self.verbose:
      self.print_settings()
    self.stop()
    self.clear()
    self.all_samples_written = False
    self.last_sample_written = 0
    self.write(2)    # Push two initial packets
    self.start(repeat=n)    
    try:
      self.write_waveform_to_fifo(n)
      if wait:
        self.wait_until_output_done()        
    except KeyboardInterrupt:
      print "\nScan interrupted by user."
      self.stop()
      self.clear()
      
  def _run_all_samples(self):
    """Write *all* samples to FIFO, start the scan, and return immediately.
    
    If the number of samples is larger than the 2kS AO FIFO, or to return
    only after the scan terminates, see :func:`_run_split_samples`."""
    assert self.nchannels*self.nsamples <= 2000, "Too many samples for FIFO! Use _run_split_samples() instead."
    if self.verbose:
      print ("%d samples writtem to AO FIFO." % (self.nchannels*self.nsamples, ))
    self.stop()
    self.clear()
    bytes_written = self._device.AOutDataToFIFO(self.data, timeout=self.timeout)
    if bytes_written/2 != self.nchannels*self.nsamples:
      raise USB2600Error("AOutDataToFIFO: %d samples send, only %d written." % (self.nchannels*self.nsamples, bytes_written/2))
    self.start(continuous=True)
    
  def run(self, n=1, wait=False, thread=False):
    """Start the scan.
    
    :param n: number of waveform repeats. 0=forever.
    :param thread: whether to run the scan in a background thread.
                   if True, call :func:`stop` to interrupt the scan,
                   other wise use :kbd:`Ctrl+C`
    :param wait: if True, block until the scan is complete.
                 If False, return as soon as all data has been written to the FIFO.
    """
    if (not wait) and self.nchannels*self.nsamples <= 2000:
      self._run_all_samples()
    elif thread:
      self.threaded = True
      self.verbose = False
      self.thread = threading.Thread(target=self._run_split_samples, args=(n, wait), name='USB2600_AO_scan')
      self.thread.start()
    else:
      self.threaded = False
      self._run_split_samples(n, wait)

  @property
  def fifo_underrun(self):
    """Return True if the AO FIFO is empty."""
    return self._device.status()['AOUT_SCAN_UNDERRUN'] == 1

  @property
  def running(self):
    """Return True while samples are being output."""
    return self._device.status()['AOUT_SCAN_RUNNING'] == 1

  @property
  def done(self):
    """Return True once all samples have been ouput."""
    return self._device.status()['AOUT_SCAN_DONE'] == 1

  #@property
  #def status(self):
  #  s = self._device.status()
  #  return s.

class AIO_Scan(object):
  """Synchronous analog input and output scan.
  
  To start the scan:

  >>> run(thread=True|False)
    * thread: whether to run in a background thread (set running=False to stop it)
      or in the main thread (Ctl+C to stop).
      
  The acquired analog input data is in the :attr:`data` attribute.
  """
  def __init__(self, device, AIchans, AOchans, AOwaveform, rate=None,
                         packet_size = 240, tf = 0.5, verbose=False):
    """   
    :param device: :class:`USB2600_Wrapper` instance
    :param AIchans: tuple of integers indicating the analog input channels.
    :param AOchans: tuple of integers indicating the analog output channels.
    :param AOwaveform: a :class:`Waveform` instance containing the analog output data.
                     Must have the same number of rows as ``AOchans``
    :param rate: sampling frequency (Hz). If not given, use ``AOwaveform.rate``.
    :param packet_size: number of bytes in each USB read/write operation.
    :param tf: USB read time factor. If getting ``AO_FIFO_UNDERRUN`` or ``AI_FIFO_OVERRUN`` errors, 
               try decreasing *tf* to pop/push data to the FIFO faster.
    :param verbose: if True, print info to console while scan is running.
    """
    # Currently retrigger has no effect, but could add support for it later
    # Assume AI throughput is larger than AO throughput, so packet_size for AO will be smaller
    ao_packet_size = packet_size * len(AOchans) / (2 * len(AIchans))
    _rate = rate if rate is not None else AOwaveform.rate
    self.aoscan = AO_Scan(device.device, AOchans, AOwaveform, _rate, 
                         trigger=True, 
                        packet_size=ao_packet_size, tf=tf, verbose=verbose)
    self.aiscan = AI_Scan(device.device, AIchans, 2 * self.aoscan.nsamples, 2 * _rate,
                         trigger=True,
                         packet_size=packet_size, tf=tf, verbose=verbose)
    self.aichans = AIchans
    self.aochans = AOchans
    self.rate = AOwaveform.rate
    self.nsamples = AOwaveform.nsamples
    self.verbose = verbose
    self._device = device
    self.__call__ = self.run

  def __repr__(self):
    return "<USB2600 synchronous analog input/output scan. AI/O channels: %(aichans)s/%(aochans)s, %(nsamples)d samples at %(rate)0.2fHz.>" % self.__dict__ 

  def print_status(self):
    """Pretty-print scan status to console."""
    ai_status = 'done' if self.aiscan.done else 'running'
    ao_status = 'done' if self.aoscan.done else 'running'
    ai_info = (ai_status, self.aiscan.last_sample_read, self.aiscan.nsamples)
    ao_info = (ao_status, self.aoscan.last_sample_written, self.aoscan.nsamples*self.aoscan.nchannels)
    self.aio_status = "AI: %s, %d/%d samples read | AO: %s, %d/%d samples written." % (ai_info + ao_info)
    if not self.threaded:
      sys.stdout.write(self.aio_status + "\r")
      sys.stdout.flush()

  def stop(self):
    """Stop the scan."""
    self.aiscan.stop()
    self.aiscan.clear()
    self.aoscan.stop()
    self.aoscan.clear()
    self.stopped = True

  def _run(self):
    # set up AO scan. Push 5 packets of AO data
    self.aoscan.stop()
    self.aoscan.clear()
    self.aoscan.last_sample_written = 0
    self.aoscan.all_samples_written = False
    self.aoscan.write(5)
    # set up AI scan
    self.aiscan.stop()
    self.aiscan.clear()
    self.aiscan.config()
    # start scans and trigger
    self._device.internal_trigger.reset()
    self.aiscan.start()
    self.aoscan.start()
    self._device.internal_trigger.trigger()
    self.stopped = False
    
    t_update = min(self.aiscan.t_update, self.aoscan.t_update)
    n_updates = min(self.aiscan.n_updates, self.aoscan.n_updates)
    sleep(3*t_update)
    
    try:
      while (not self.stopped) and (not self.aiscan.all_samples_read):
        sleep(self.aiscan.tf * t_update)
        if not self.aoscan.all_samples_written: # Push AO data 
          self.aoscan.write()
        if not self.aiscan.all_samples_read: # Pop AI data
          self.aiscan.read()
        if not self.threaded:
          self.print_status()
        if self.aiscan.fifo_overrun:
          raise USB2600Error('Analog input FIFO overrun! ')
        if self.aoscan.fifo_underrun:
          raise USB2600Error('Analog output FIFO underrun! ')
    except KeyboardInterrupt:
      print "\nScan interrupted by user."
      self.stop()
      
    # Package data as waveform, keeping only the relevant parts, and copy attributes
    self.data = waveform.Waveform(self.aiscan.data[:,1:2*self.aoscan.nsamples:2],
                                  dt =1/self.aoscan.rate)
    try:
      self.data.linear_subset = self.aoscan.waveform.linear_subset # copy original linear_subset, if present
    except AttributeError:
      pass

  def run(self, thread=False):
    """Start the scan.
    
    :param thread: whether to run the scan in a background thread.
                   if True, set :attr:`running` = False to interrupt the scan,
                   otherwise use :kbd:`Ctrl+C`
    """
    if thread:
      self.threaded = True
      self.verbose = False
      self.thread = threading.Thread(target=self._run, args=(), name='USB2600_SyncAIO')
      self.thread.start()
    else:
      self.threaded = False
      self._run()



# broken attempt at using property descriptors for DIO

#class DIO_Port_Dir(object):
	#def __init__(self, device, port, direction):
		#"""
		#:param device: :class:`USB2600_wrapper`
		#:param int port: port number
		#:param bool direction: True for input, False for output 
		#"""
		#self._device = device
		#self._port = port
		#self._dir = direction
		##print "Port_Dir " + str(self._port) + ", dir=" + str(self._dir)
	
	#def __get__(self, obj, objtype=None):
		#byte_in = obj._device.dioConfig(self._port)
		##print self._dir
		##print byte_in
		#return (byte_in if obj._dir else (~byte_in & 0xFF))
		
	#def __set__(self, obj, byte):
		#byte_out = (byte if self._dir else (~byte & 0xFF))
		##print byte_out
		#self._device.dioConfig(self._port, byte_out)

#class DIO_Pin_Dir(object):
	#def __init__(self, device, port, pin, direction):
		#"""
		#:param device: :class:`USB2600_wrapper`
		#:param int port: port number
		#:param int pin: pin number
		#:param bool direction: True for input, False for output 
		#"""
		#self._device = device
		#self._port = port
		#self._pin = pin
		#self._dir = direction

	#def __get__(self, obj, objtype=None):
		#"""Query whether the pin is configured as input, or configure it so by setting it to True."""
		#print self._port, self._pin, self._dir, bin(self._device.dioConfig(self._port))
		#is_input = bool((self._device.dioConfig(self._port) >> self._pin) & 1)
		#print is_input
		##print is_input
		#return (is_input if self._dir else not is_input)

	#def __set__(self, obj, boolean):
		##byte_out = self._device.dioConfig(self._port) | ((boolean if self._dir else not boolean) << self._pin)
		##print bin(byte_out)
		#current = self._device.dioConfig(self._port)
		#bit_out = boolean if self._dir else not boolean
		#byte_out = current ^ ((not(bit_out) ^ current) & (1 << self._pin))
		#self._device.dioConfig(self._port, byte_out)


#class DIO_Port2(object):
  #"""Property descriptor for digital input-output ports."""
  #def __init__(self, device, port):
    #"""
    #:param device: the :class:`USB2600_Wrapper` instance
    #:param int port: port index on the device
    #"""
    #self._device = device
    #self._port = port
    
  #def __get__(self, obj, objtype=None):
    #"""Read or query the port state.
    
    #:rtype: int
    
    #.. Tip:: use bin() for a binary representation of the returned value.
    #"""
    #return self._device.dioRead(self._port)
      
  #def __set__(self, obj, byte):
    #self._device.dioWrite(self._port, byte)


#class DIO_Pin2(object):
  #"""Property descriptor for individual pins of the digital input-output ports."""
  #def __init__(self, port, pin):
    #"""
    #:param port: a :class:`DIO_Port` instance
    #:param pin: the target pin on ``port``
    #"""
    #setattr(self.__class__, '_port', port)
    #self._pin = pin
   
  #def __set__(self, obj, boolean):
    #"""Set or query the pin state."""
    #self._port = self._port | (boolean << self._pin)

  #def __get__(self, obj, objtype=None):
    #return bool((self._port >> self._pin) & 1)
    
  #def config(self):
    #print "Configuring!"
    

### OLD DIO

#class DIO_Port(object):
  #""" Each digital port is represented by one instance of this class.
  
  #:Usage:
  
  #>>> daq = usb2600.USB2600()
  #>>> daq.dioA.config('input')  # set all pins on port A as inputs
  #>>> daq.dioB.config('output') #                      B    outputs
  #>>> daq.dioB.write(1)         # pin 0 high
  #>>> daq.dioB.write(2)         # pin 1 high, 0 low
  #>>> daq.dioB.write(3)         # pin 0 and 1 high
  #>>> daq.dioB.write(0)         # all pins low
  
  #.. Tip:: 
     #Pull-up configuration (from doc): 
     #"Each port has 47 kohms resistors that are configurable as pull-up or pull-down (default)
     #using an onboard jumper (W5, W6, W7)."
  
  #"""
  #def __init__(self, device, port):
    #"""
    #:param device: the :class:`USB2600` instance
    #:param port: port index on the device
    #:type port: int
    #"""
    #self._device = device
    #self._port = port
    ##self.config('input')
    
  #def config(self, direction):
    #"""Configure the port as output or input.
    
    #:param direction:
    
       #- a string: 'input' or 'output' will set all the pins on the port
         #as outputs or inputs
       #- an integer: to mix inputs and outputs on the same port,
         #set bit n to 1 for input, 0 for output.
    #:type direction: int or string
    #"""
    #if direction == 'output':
      #self._direction = 0
      #self._device.dioConfig(self._port, 0)
    #elif direction == 'input':
      #self._direction = 0xFF
      #self._device.dioConfig(self._port, 0xFF)
    #else:
      #self._direction = direction
      #self._device.dioConfig(self._port, direction)

  #@property
  #def direction(self):
    #"""Query or configure the port direction (0xFF all inputs, 0x00 all ouputs, or a mixture: 0x01, etc.)"""
    #return self._direction
  #@direction.setter
  #def direction(self, byte):
    #self._direction = byte
    #self._device.dioConfig(self._port, direction)
    
  #def read(self):
    #"""Read the port state.
    
    #Convert the returned integer to binary to identify the state of the
    #individual pins.
    
    #:rtype: int
    #"""
    #return self._device.dioRead(self._port)
      
  #def write(self, byte):
    #"""Write to the port.
    
    #The relevant pins must have been configured as output."""
    #self._device.dioWrite(self._port, byte)
  
  #def __call__(self, byte=None):
    #if byte is None:
      #return self.read()
    #else:
      #self.write(byte)


### OLD TRIGGER

#class _Trigger(object):
  #"""Represents a trigger.
  
  #Defaults to an external trigger on a rising edge.
  
  #To make an internal/software trigger, specify a digital port and pin
  #(which must be connected to the trigger input on the board).
  #"""
  #def __init__(self, device, port=None, pin=None, diopin=None, retrigger=1, mode='Edge', polarity='Rising'):
    #"""
    #:param mode: 'Edge' or 'Level'.
    #:param polarity: in 'Edge' mode:  'Rising' or 'Falling'.
                     #In 'Level' mode : 'Low' or 'High'.
    #:param retrigger: NOT IMPLEMENTED
    #:param port: a :class:`DIO_Port` instance
    #:param pin: a pin on this port.
    
    #If ``port`` and ``pin`` are specified, use as an internal trigger.
                           #If not, it's an external trigger
    #"""
    #self._device = device
    #self.port = port
    #self.pin = pin
    #self.retrigger = retrigger
    #self.mode_string = mode
    #self.polarity_string = polarity
    #if mode == 'Edge':
      #self.mode = 1
    #elif mode == 'Level':
      #self.mode = 0
    #else:
      #raise ValueError('Trigger mode must be "Edge" or "Level".')
    #if polarity in ('Rising', 'High'): 
      #self.polarity = 1
    #elif polarity in ('Falling', 'Low'):
      #self.polarity = 0
    #else:
      #raise ValueError('Trigger polarity should be "Rising" or "Falling", or "High" or "Low".')
    #self.config()
    
  #def __repr__(self):
    #return "USB2600 trigger: " + self.polarity_string + ' ' + self.mode_string
    
  #def config(self):
    #"""Apply configuration. 
    
    #This is normally done automatically when creating the Trigger instance."""
    #self._device.TriggerConfig(mode = self.mode_string, polarity = self.polarity_string)
    #if self.port is not None:
      #self.port.config('output')
    
  #def trigger(self):
    #"""Fire the trigger."""
    #assert self.port is not None, 'Internal trigger port not assigned'
    #assert self.pin is not None, 'Internal trigger pin not assigned'
    #self.port.write(self.polarity << self.pin)
    
  #def reset(self):
    #"""Reset the trigger so it can be fired again."""
    ## initialise digital output to 0 (Low) if polarity is 1 (Rising)
    ##                        or to 1 (High)               0 (Falling)
    #assert self.port is not None, 'Internal trigger not assigned'
    #self.port.write(1-self.polarity) 
    
## Consider a ContextManager to avoid reconfiguring the trigger in SyncAIO?
