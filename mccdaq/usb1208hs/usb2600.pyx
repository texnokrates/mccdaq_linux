#cython: embedsignature=True
"""Linux Cython driver for the MCCDAQ USB-26xx series data acquisition devices.

This is the low-level Cython module, appearing as the ``wrapper`` module 
in the mccdaq/usb2600 package. It is mainly a direct wrapper to
functions in usb-2600.h, with some minor changes to make it more user-
friendly.

It is recommanded to use the high-level interface in the :mod:`mccdaq.usb2600.usb2600` module.

Warren Jasper's original C driver can be found at
ftp://lx10.tx.ncsu.edu/pub/Linux/drivers/USB

(c) 2014-15 Guillaume Lepert, Imperial College London.

"""
 
# 26 Aug 2015: Rename module as usb2600.pyx. Move C definition to pxd file.
#              Move pure python code to usb2600.py
#              Remove AInLongWaveform, AOutLongWaveform and snap functions
#              (use AIScan, AOScan and synchronousAIOScan from usb2600.py instead).
# 24 Sep 2015: minor clean-up (replace for loops with list comprehensions,
#              remove perror/printf, add Exceptions.
 
import numpy as np
cimport numpy as np
np.import_array()

cimport cython
from usb2600 cimport *

usb1208hs_model = {'1208HS': USB1208HS_PID, '1208HS_2AO': USB1208HS_2AO_PID, '1208HS_4AO': USB1208HS_4AO_PID}

cdef class USB1208HS_Wrapper:
  """Low-level class for controlling USB-2600 series DAQ devices from MCC."""
  cdef usb_dev_handle *usb_handle
  cdef np.ndarray table_AI
  cdef np.ndarray table_AO
  cdef ScanList *scanList
  port = {'A':0, 'a':0, 0:0, 'B':1, 'b':1, 1:1, 'C':2, 'c':2, 2:2}      # DIO port names lookup: can refer to port A as 'A', 'a' or 0
  trigger_mode = {'Edge':1, 'Level':0}
  trigger_polarity = {'Rising':1, 'Falling':0}
  #last_ai_config = ()
   
  @cython.boundscheck(False)
  @cython.wraparound(False)
  def __cinit__(self, model='12O8HS'):
    """
    :param model: USB-1208HS model ('1208HS', '1208HS_2AO', or '1208HS_4AO')
    :type  model: str"""
    self.model = model
    self.usb_handle=usb_device_find_USB_MCC(usb1208hs_model[model])
    usbInit_1208HS(self.usb_handle)
    # Python 'float' is actually a C 'double', ie 64bits, whereas the BuildGainTable
    # function take a C 'float, ie 32bits. So use the 32 bits numpy type otherwise the arrays will not have the required size!  
    cdef np.ndarray[np.float32_t, mode="c",ndim=2] tAI = \
      np.ascontiguousarray(np.empty(shape=(NGAINS_1208HS,2),dtype=np.float32))
    cdef np.ndarray[np.float32_t, mode="c",ndim=2] tAO = \
      np.ascontiguousarray(np.empty(shape=(NCHAN_AO_1208HS,2),dtype=np.float32))
    usbBuildGainTable_USB1208HS(self.usb_handle, &tAI[0,0]);
    self.table_AI = tAI
    if model == '1208HS_2AO' or model == '1208_4AO':
        usbBuildGainTable_USB1208HS_4AO(self.usb_handle, &tAO[0,0]);
        self.table_AO = tAO    
    self.scanList = makeScanList()
    
  def print_calibration_table(self):
    cdef np.ndarray[np.float32_t, mode="c",ndim=2] tAI = \
      np.ascontiguousarray(self.table_AI, dtype=np.float32)
      
    cdef np.ndarray[np.float32_t, mode="c",ndim=2] tAO 
    
    if model == '1208HS_2AO' or model == '1208HS_4AO':
      tAO = np.ascontiguousarray(self.table_AO, dtype=np.float32)
    print("\nAnalog Input Calibration Table\n");
    for i in range(NGAINS_1208HS):
      print("Gain: %d   Slope = %f   Offset = %f\n", i, tAI[i,0], tAI[i,1])
    if model == '1208HS_2AO' or model == '1208HS_4AO':
        print("\nAnalog Output Calibration Table\n");
        for i in range(NGAINS_1208HS):
          print("Channel: %d   Slope = %f   Offset = %f\n", i, tAO[i,0], tAO[i,1])
    
  def __str__(self):
    return "<MCC USB-1208HS (low-level interface)>"
  
  def __del__(self):
    print("Closing device.")
    cleanup_USB1208HS(self.usb_handle)
    freeScanList(self.scanList)
    
  cdef float AItoVolts(self, voltage):
    gain = 0
    return volts_USB1208HS(self.usb_handle, gain, 
                         round(voltage*self.table_AI[gain,0] + self.table_AI[gain,1]))

  def AIn(self, channel):
    """Single sample, single channel analog input."""
    gain = 0
    updateScanList(self.scanList, 0, (LAST_CHANNEL | SINGLE_ENDED), gain, channel)
    #self.last_ai_config = (channel,)
    usbAInConfig_USB1208HS(self.usb_handle, self.scanList)
    
    return self.AItoVolts(usbAIn_USB1208HS(self.usb_handle, channel))

  def AOut(self, channel, voltage):
    """Single channel, single sample analog output."""
    cdef np.ndarray[np.float32_t, mode="c",ndim=2] tAO = self.table_AO
    usbAOut_USB26X7(self.usb_handle, channel, voltage, &tAO[0,0])
    
  def AOutRead(self, channel):
    cdef double voltage
    cdef np.ndarray[np.float32_t, mode="c",ndim=2] tAO = self.table_AO
    usbAOutR_USB26X7(self.usb_handle, channel, &voltage, &tAO[0,0])
    return voltage
      
  def AOutScanStop(self):
    """Interrupts a analog output scan."""
    usbAOutScanStop_USB26X7(self.usb_handle)
    
  def AOutScanClear(self):
    """Clears the AO FIFO."""
    usbAOutScanClearFIFO_USB26X7(self.usb_handle)
    
  def AOutScanStart(self, channels, rate,
                   nsamples=0, trigger=False, retrigger=1):
    """Start a multi channel analog out scan. Data must have been loaded to FIFO.
    Rate in Hz. Set rate = 0 to use the external clock (AO_CLK_IN).
    If nsamples = n > 0, only n samples PER CHANNEL are pushed to the FIFO.
    If nsamples = 0 (default), the scan runs continuously until the FIFO is empty or the scan stopped.
    Set trigger = True to start scanning on external trigger XTTLTRIG, and
    use retrigger = n (n>1) to repeat over n extrenal triggers.
    """
    if rate == 0:       # for external clock
      Rate = 64.E6
    else:
      Rate = rate
    cdef np.uint8_t chan = 0
    for c in channels:
      chan = chan | (0x1 << c)      
    cdef np.uint8_t options = chan | (trigger << 4) | ((retrigger>1) << 5)
    usbAOutScanStart_USB26X7(self.usb_handle, nsamples*retrigger, nsamples, rate, options)
  
  def AOutDataToFIFO(self, np.ndarray[np.uint16_t] data, timeout=1000):
    """Transfer AO data to the AO FIFO.
    
    Data should have been processed by AOoutData() first
    to calibrate the input (float) waveform and convert it to 16-bits int.
    """
    cdef np.ndarray[np.uint16_t, mode="c", ndim=1] wf = np.ascontiguousarray(data)
    bytes_written = usb_bulk_write(self.usb_handle, USB_ENDPOINT_OUT|2, <char *>&wf[0], data.size * 2, timeout)
    if bytes_written < 0:
      raise RuntimeError("usb_bulk_write error in AOutDataToFIFO")
    return bytes_written
    #else:
    #  print('Written '+ str(bytes_written) + ' bytes to AO FIFO.')
        
  def AOutData(self, voltages, channel=0):
    """Convert a list or array of floats to its 16-bits, calibrated binary representation
    for analog output.
    If channel is not specified, takes the calibration from channel 0."""
    converted = [(v/10.*32768. + 32768.)*self.table_AO[channel,0]+self.table_AO[channel,1] for v in voltages]
    cdef np.ndarray[np.uint16_t, mode="c", ndim=1] data = np.ascontiguousarray(converted, dtype=np.uint16)
    return data
  
  def AInConfig(self, channels):
    """Configure the AI channels scan list. CHANNELS must be a tuple, even if configuring a single channel."""
    gain=0
    for i, ch in enumerate(channels):
      updateScanList(self.scanList, i, SINGLE_ENDED, gain, ch)
    updateScanList(self.scanList, len(channels)-1, (LAST_CHANNEL | SINGLE_ENDED), gain, channels[-1])
    usbAInConfig_USB1208HS(self.usb_handle, self.scanList)

  def AInScanStart(self, channels, nsamples, rate,
                   burst=False, trigger=False, retrigger=1, packet_size=0xff):
    """Start the analog input scan to acquire nsamples.
    Channels is a list/tuple of channels to scan in the indicated order.
    Set burst=True for Burst mode, trigger=True to use external trigger on XTTLTRIG,
    In triggered mode, set retrigger=n (n>=1) to repeat the acquistion of nsamples n times.
    Set rate=0Hz to use external clock on AI_CLK_IN.
    """
    nchan=len(channels)
    gain=0
    if rate == 0:       # for external clock
      Rate = 64.E6
    else:
      Rate = rate
    cdef np.uint8_t options = ((burst << 0) | (trigger << 3) | ((retrigger>1) << 6))
    usbAInScanStart_USB1208HS(self.usb_handle, nsamples*retrigger, nsamples, rate, packet_size, options);
    
  def AInScanRead(self, nchannels, nsamples):
    """Read nchannels x nsamples of AI FIFO data. """
    cdef np.ndarray[np.uint16_t, mode="c", ndim=1] data = \
      np.ascontiguousarray(np.empty(shape=nsamples*nchannels, dtype=np.uint16))
    bytes_read = usbAInScanRead_USB1208HS(self.usb_handle, nsamples, nchannels, &data[0])
    if bytes_read < 0:
      raise RuntimeError('Error in AInScanRead! ')
    data_in_volts = [self.AItoVolts(d) for d in data]
    return (np.reshape(data_in_volts, (nchannels, nsamples), 'F'), bytes_read)
  
  def AInScanStop(self):
    """Stop the AI scan. """ 
    usbAInScanStop_USB1208HS(self.usb_handle)
    
  def AInScanClear(self):
    """ Clear the AI FIFO. Data will be lost! """
    usbAInScanClearFIFO_USB1208HS(self.usb_handle)
  
  def TriggerConfig(self, action='Set', mode='Edge', polarity='Rising'):
    """Configure the analog input trigger. 
       options: mode:     'Edge' or 'Level'
                polarity: 'Rising'or 'Falling'
       AInScan must be configured to use the trigger.
       If first argument action is set, will query the current configuration
       instead of setting it (eg AInTriggerConfig('?'))."""
    cdef int options
    if action == 'Set':
      usbTriggerConfig_USB1208HS(self.usb_handle, self.trigger_mode[mode] + (self.trigger_polarity[polarity] << 1))
    else:
      trigger_polarity = dict(zip(self.trigger_polarity.values(),self.trigger_polarity))
      trigger_mode = dict(zip(self.trigger_mode.values(),self.trigger_mode))
      usbTriggerConfigR_USB1208HS(self.usb_handle, &options)
      print('AI trigger currently congigured as: mode = ' + trigger_mode[(options & 1)] + ', polarity = ' + trigger_polarity[(options & 2) >>1])
    
  def blink(self, count=1):
    cdef int c = count
    usbBlink_USB1208HS(self.usb_handle, c)
    
  def temperature(self):
    cdef float temp
    usbTemperature_USB1208HS(self.usb_handle, &temp)
    return temp
  
  def serialNumber(self):
    cdef char *ser
    ser = "123465789"
    usbGetSerialNumber_USB1208HS(self.usb_handle, ser)
    return ser
  
  def status(self):
    """Returns the device status byte."""
    status_byte = usbStatus_USB1208HS(self.usb_handle)
    #print("Status = %#x\n", status_byte)
    return {'AIN_SCAN_RUNNING':((status_byte & AIN_SCAN_RUNNING) >> 1),
            'AIN_SCAN_OVERRUN':((status_byte & AIN_SCAN_OVERRUN) >> 2),
            'AOUT_SCAN_RUNNING':((status_byte & AOUT_SCAN_RUNNING) >> 3),
            'AOUT_SCAN_UNDERRUN':((status_byte & AOUT_SCAN_UNDERRUN) >> 4),
            'AIN_SCAN_DONE':((status_byte & AIN_SCAN_DONE) >> 5),
            'AOUT_SCAN_DONE':((status_byte & AOUT_SCAN_DONE) >> 6),
            'FPGA_CONFIGURED':((status_byte & FPGA_CONFIGURED) >> 8),
            'FPGA_CONFIG_MODE':((status_byte & FPGA_CONFIG_MODE) >> 9)}
 
  def FPGAVersion(self):
    cdef int version
    usbFPGAVersion_USB1208HS(self.usb_handle, &version)
    return version
 
  def reset(self):
    """Causes the device to perform a reset, disconnects from the USB bus and resets its microcontroller."""
    usbReset_USB1208HS(self.usb_handle)
  
  def counterInit(self, counter):
    usbCounterInit_USB1208HS(self.usb_handle, counter)
    
  def counterRead(self, counter):
    return usbCounter_USB1208HS(self.usb_handle, counter)
  
  def dioConfig(self, port, value=None):
    """Set or query the direction of a tristate register.
    
    :param port: the DIO port index (can be 'A', 'a', or 0)
    :param int value: 8-bit integer (1 for input, 0 for output).
                      If None, query the current configuration.
    """
    if value is None:
      return usbDTristateR_USB1208HS(self.usb_handle, self.port[port]) 
    else:     
      usbDTristateW_USB1208HS(self.usb_handle, self.port[port], value)
    
  def dioWrite(self, port, value):
    usbDLatchW_USB1208HS(self.usb_handle, self.port[port], value)
  
  def dioRead(self, port):
    return usbDLatchR_USB1208HS(self.usb_handle, self.port[port])
  
  def timerControlRead(self, timer):
    cdef int control = 0
    usbTimerControlR_USB1208HS(self.usb_handle, timer, &control)
    return {'Enable' : bool(control & 1),
            'Running': bool((control >> 1) & 1),
            'Invert' : bool((control >> 2) & 1)}

  def timerControlWrite(self, timer, enable, invert=False):
    cdef int control = ((enable) | (invert << 2))
    usbTimerControlW_USB1208HS(self.usb_handle, timer, control)

  def timerPeriodWrite(self, timer, period):
    """ frequency = 64 MHz / (period + 1) """
    usbTimerPeriodW_USB1208HS(self.usb_handle, timer, period)
  
  def timerPeriodRead(self, timer):
    cdef int period = 0
    usbTimerPeriodR_USB1208HS(self.usb_handle, timer, &period)
    return period
  
  def timerPulseWidthWrite(self, timer, pulse_width):
    usbTimerPulseWidthW_USB1208HS(self.usb_handle, timer, pulse_width)
  
  def timerPulseWidthRead(self, timer):
    cdef int pw = 0
    usbTimerPulseWidthR_USB1208HS(self.usb_handle, timer, &pw)
    return pw
  
  def timerCountRead(self, timer):
    cdef int count = 0
    usbTimerCountR_USB1208HS(self.usb_handle, timer, &count)
    return count
  
  def timerCountWrite(self, timer, count):
    usbTimerCountW_USB1208HS(self.usb_handle, timer, count)
    
  def timerDelayRead(self, timer):
    cdef int delay = 0
    usbTimerDelayR_USB1208HS(self.usb_handle, timer, &delay)
    return delay
  
  def timerDelayWrite(self, timer, delay):
    usbTimerDelayW_USB1208HS(self.usb_handle, timer, delay)    
    
    

