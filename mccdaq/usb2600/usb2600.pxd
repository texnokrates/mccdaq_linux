# Definition file for MCCDAQ USB2600 Linux driver
# Gets definitions from pmd.h, usb-26xx.rbf, usb-2600.h, and mccdaq-extra.h

import numpy as np
cimport numpy as np

# Declare C headers to cython compiler
cdef extern from "<stdlib.h>":
  pass
cdef extern from "<stdio.h>":
  pass
cdef extern from "<string.h>":
  pass
cdef extern from "<sys/ioctl.h>":
  pass
cdef extern from "<sys/types.h>":
  pass
cdef extern from "<sys/stat.h>":
  pass
cdef extern from "<asm/types.h>":
  pass
cdef extern from "<fcntl.h>":
  pass
cdef extern from "<unistd.h>":
  pass
cdef extern from "<errno.h>":
  pass
cdef extern from "<math.h>":
  pass
cdef extern from "<string.h>":
  pass
  
cdef extern from "<usb.h>":
  # we'll need direct interaction with the device via libusb
  struct usb_dev_handle:
    pass
  int usb_bulk_write(usb_dev_handle *dev, int ep, char *bytes, int size, int timeout)
  int USB_ENDPOINT_IN
  int USB_ENDPOINT_OUT

cdef extern from "pmd.h":
  usb_dev_handle* usb_device_find_USB_MCC(int productId)
  int usb_get_max_packet_size(usb_dev_handle* udev, int endpointNum)
  
cdef extern from "usb-26xx.rbf":
  char* FPGA_data
  
cdef extern from "usb-2600.h":
  # Import #define constants
  int  USB2623_PID
  int  USB2627_PID
  int  USB2633_PID
  int  USB2637_PID
  
  int  HOST_TO_DEVICE
  int  DEVICE_TO_HOST
  int  STANDARD_TYPE
  int  CLASS_TYPE
  int  VENDOR_TYPE
  int  RESERVED_TYPE
  int  DEVICE_RECIPIENT
  int  INTERFACE_RECIPIENT
  int  ENDPOINT_RECIPIENT
  int  OTHER_RECIPIENT
  int  RESERVED_RECIPIENT
  
  int  DTRISTATE
  int  DPORT
  int  DLATCH
  
  int  AIN
  int  AIN_SCAN_START
  int  AIN_SCAN_STOP
  int  AIN_CONFIG
  int  AIN_CLR_FIFO
  
  int  AOUT
  int  AOUT_SCAN_START
  int  AOUT_SCAN_STOP
  int  AOUT_CLEAR_FIFO
  
  int  COUNTER
  int  TIMER_CONTROL
  int  TIMER_PERIOD
  int  TIMER_PULSE_WIDTH
  int  TIMER_COUNT
  int  TIMER_START_DELAY
  int  TIMER_PARAMETERS
  
  int  MEMORY
  int  MEM_ADDRESS
  int  MEM_WRITE_ENABLE
  
  int  STATUS
  int  BLINK_LED
  int  RESET
  int  TRIGGER_CONFIG
  int  CAL_CONFIG
  int  TEMPERATURE
  int  SERIAL
  
  int  FPGA_CONFIG
  int  FPGA_DATA
  int  FPGA_VERSION
  
  int  COUNTER0
  int  COUNTER1
  int  COUNTER2
  int  COUNTER3
  int  TIMER0
  int  TIMER1
  int  TIMER2
  int  TIMER3
  
  int  SINGLE_ENDED
  int  CALIBRATION
  int  LAST_CHANNEL
  int  PACKET_SIZE
  
  int  BP_10V
  int  BP_5V
  int  BP_2V
  int  BP_1V
  
  int  AO_CHAN0
  int  AO_CHAN1
  int  AO_CHAN2
  int  AO_CHAN3
  int  AO_TRIG
  int  AO_RETRIG_MODE
  
  int  AIN_SCAN_RUNNING
  int  AIN_SCAN_OVERRUN
  int  AOUT_SCAN_RUNNING
  int  AOUT_SCAN_UNDERRUN
  int  AIN_SCAN_DONE
  int  AOUT_SCAN_DONE
  int  FPGA_CONFIGURED
  int  FPGA_CONFIG_MODE
  
  int  NCHAN_2600
  int  NGAINS_2600
  int  NCHAN_AO_26X7
  int  MAX_PACKET_SIZE_HS
  int  MAX_PACKET_SIZE_FS

  ctypedef struct timerParams:
    int period
    int pulseWidth
    int count
    int delay
    
  ctypedef struct ScanList:
    int mode
    int range
    int channel
  
  # Import functions
  # __uxx types have been replaced by int  
  void usbDTristateW_USB2600(usb_dev_handle *udev, int port, int value)
  int usbDTristateR_USB2600(usb_dev_handle *udev, int port)
  int usbDPort_USB2600(usb_dev_handle *udev, int port)
  void usbDLatchW_USB2600(usb_dev_handle *udev,int port,  int value)
  int usbDLatchR_USB2600(usb_dev_handle *udev, int port)
  void usbBlink_USB2600(usb_dev_handle *udev, int count)
  void cleanup_USB2600( usb_dev_handle *udev)
  void usbTemperature_USB2600(usb_dev_handle *udev, float *temperature)
  void usbGetSerialNumber_USB2600(usb_dev_handle *udev, char serial[9])
  void usbReset_USB2600(usb_dev_handle *udev)
  void usbFPGAConfig_USB2600(usb_dev_handle *udev)
  void usbFPGAData_USB2600(usb_dev_handle *udev, int *data, int length)
  void usbFPGAVersion_USB2600(usb_dev_handle *udev, int *version)
  int usbStatus_USB2600(usb_dev_handle *udev)
  void usbInit_2600(usb_dev_handle *udev)
  void usbCounterInit_USB2600(usb_dev_handle *udev, int counter)
  int usbCounter_USB2600(usb_dev_handle *udev, int counter)
  void usbTimerControlR_USB2600(usb_dev_handle *udev, int timer, int *control)
  void usbTimerControlW_USB2600(usb_dev_handle *udev, int timer, int control)
  void usbTimerPeriodR_USB2600(usb_dev_handle *udev, int timer, int *period)
  void usbTimerPeriodW_USB2600(usb_dev_handle *udev, int timer, int period)
  void usbTimerPulseWidthR_USB2600(usb_dev_handle *udev, int timer, int *pulseWidth)
  void usbTimerPulseWidthW_USB2600(usb_dev_handle *udev, int timer, int pulseWidth)
  void usbTimerCountR_USB2600(usb_dev_handle *udev, int timer, int *count)
  void usbTimerCountW_USB2600(usb_dev_handle *udev, int timer, int count)
  void usbTimerDelayR_USB2600(usb_dev_handle *udev, int timer, int *delay)
  void usbTimerDelayW_USB2600(usb_dev_handle *udev, int timer, int delay)
  void usbTimerParamsR_USB2600(usb_dev_handle *udev, int timer, timerParams *params)
  void usbTimerParamsW_USB2600(usb_dev_handle *udev, int timer, timerParams *params)
  void usbMemoryR_USB2600(usb_dev_handle *udev, int *data, int length)
  void usbMemoryW_USB2600(usb_dev_handle *udev, int *data, int length)
  void usbMemAddressR_USB2600(usb_dev_handle *udev, int address)
  void usbMemAddressW_USB2600(usb_dev_handle *udev, int address)
  void usbMemWriteEnable_USB2600(usb_dev_handle *udev)
  void usbTriggerConfig_USB2600(usb_dev_handle *udev, int options)
  void usbTriggerConfigR_USB2600(usb_dev_handle *udev, int *options)
  void usbTemperature_USB2600(usb_dev_handle *udev, float *temperature)
  void usbGetSerialNumber_USB2600(usb_dev_handle *udev, char serial[9])
  int usbAIn_USB2600(usb_dev_handle *udev, int channel)
  void usbAInScanStart_USB2600(usb_dev_handle *udev, int count, int retrig_count, double frequency, int packet_size, int options)
  void usbAInScanStop_USB2600(usb_dev_handle *udev)
  # use replaced __uint16 by numpy 16 bit unsigned int to get data
  int usbAInScanRead_USB2600(usb_dev_handle *udev, int nScan, int nChan, np.uint16_t *data)
  void usbAInConfig_USB2600(usb_dev_handle *udev, ScanList *scanList)
  void usbAInConfigR_USB2600(usb_dev_handle *udev, ScanList *scanList)
  void usbAInScanClearFIFO_USB2600(usb_dev_handle *udev)
  void usbAOut_USB26X7(usb_dev_handle *udev, int channel, double voltage, np.float32_t *table_AO)
  void usbAOutR_USB26X7(usb_dev_handle *udev, int channel, double *voltage, np.float32_t *table_AO)
  void usbAOutScanStop_USB26X7(usb_dev_handle *udev)
  void usbAOutScanClearFIFO_USB26X7(usb_dev_handle *udev)
  void usbAOutScanStart_USB2600(usb_dev_handle *udev, int count, int retrig_count, double frequency, int options)
  # use 32 bit float for the calibration tables
  void usbBuildGainTable_USB2600(usb_dev_handle *udev, np.float32_t *table_AIN)
  void usbBuildGainTable_USB26X7(usb_dev_handle *udev, np.float32_t *table_AO)
  void usbAOutScanStart_USB26X7(usb_dev_handle *udev, int count, int retrig_count, double frequency, int options)
  double volts_USB2600(usb_dev_handle *udev, int gain, int value)
  
cdef extern from "mccdaq-extra.h":
  # Additional C functions to avoid manipulating ScanList object in Python world.
  ScanList *makeScanList()
  void freeScanList(ScanList *list)
  void updateScanList(ScanList *list, int item, int mode, int range, int channel)
