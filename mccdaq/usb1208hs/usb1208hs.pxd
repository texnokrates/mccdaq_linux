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
  
cdef extern from "usb-1208HS.rbf":
  char* FPGA_data
  
cdef extern from "usb-1208HS.h":
  int USB1208HS_PID     
  int USB1208HS_2AO_PID 
  int USB1208HS_4AO_PID 
  
  # Status bit values 
  int AIN_SCAN_RUNNING   
  int AIN_SCAN_OVERRUN   
  int AOUT_SCAN_RUNNING  
  int AOUT_SCAN_UNDERRUN 
  int AIN_SCAN_DONE      
  int AOUT_SCAN_DONE     
  int FPGA_CONFIGURED    
  int FPGA_CONFIG_MODE   
  
  # Counter Timer 
  int COUNTER0          #  Counter 0
  int COUNTER1          #  Counter 1
  
  int NCHAN_1208HS         # max number of A/D channels in the device
  int NGAINS_1208HS        # max number of gain levels (analog input)
  int NMODE                # max number of configuration modes
  int NCHAN_AO_1208HS      # number of analog output channels
  int MAX_PACKET_SIZE_HS   # max packet size for HS device
  int MAX_PACKET_SIZE_FS   # max packet size for FS device
  
  
  # Analog Input Scan and Modes 
  int SINGLE_ENDED           # 8 single-ended inputs
  int PSEUDO_DIFFERENTIAL    # 4 pseudo differential inputs
  int DIFFERENTIAL           #/g 4 true differential inputs
  int PSEUDO_DIFFERENTIAL_UP # 7 pseudo differential inputs
  int PACKET_SIZE            # max bulk transfer size in bytes
  
  # Analog Input Scan Options 
  int CHAN0 
  int CHAN1 
  int CHAN2 
  int CHAN3 
  int CHAN4 
  int CHAN5 
  int CHAN6 
  int CHAN7 
  
  int BURST_MODE   
  int CONTINUOUS   
  int TRIGGER_MODE 
  int DEBUG_MODE   
  int RETRIG_MODE  
  
  int BP_10V   
  int BP_5V    
  int BP_2_5V  
  int UP_10V   
  
  int BP_20V_DE 
  int BP_10V_DE 
  int BP_5V_DE  
  
  # Ananlog Output Scan Options
  int AO_CHAN0         # Include Channel 0 in output scan
  int AO_CHAN1         # Include Channel 1 in output scan
  int AO_CHAN2         # Include Channel 2 in output scan
  int AO_CHAN3         # Include Channel 3 in output scan
  int AO_TRIG          # Use Trigger
  int AO_RETRIG_MODE   # Retrigger Mode
  
  ctypedef struct timerParams_t {
    uint32_t period;
    uint32_t pulseWidth;
    uint32_t count;
    uint32_t delay;
  } timerParams;
  
  # function prototypes for the USB-1208HS 
  # FIXME should I replace uint types with int, as in with usb2600?
  uint16_t  usbDTristateR_USB1208HS(libusb_device_handle *udev);
  void usbDTristateW_USB1208HS(libusb_device_handle *udev, uint16_t value);
  uint16_t usbDPort_USB1208HS(libusb_device_handle *udev);
  uint16_t usbDLatchR_USB1208HS(libusb_device_handle *udev);
  void usbDLatchW_USB1208HS(libusb_device_handle *udev, uint16_t data);
  void cleanup_USB1208HS(libusb_device_handle *udev);
  void usbBlink_USB1208HS(libusb_device_handle *udev, uint8_t count);
  void usbTemperature_USB1208HS(libusb_device_handle *udev, float *temperature);
  void usbGetSerialNumber_USB1208HS(libusb_device_handle *udev, char serial[9]);
  void usbReset_USB1208HS(libusb_device_handle *udev);
  void usbFPGAConfig_USB1208HS(libusb_device_handle *udev);
  void usbFPGAData_USB1208HS(libusb_device_handle *udev, uint8_t *data, uint8_t length);
  void usbFPGAVersion_USB1208HS(libusb_device_handle *udev, uint16_t *version);
  uint16_t usbStatus_USB1208HS(libusb_device_handle *udev);
  void usbMemoryR_USB1208HS(libusb_device_handle *udev, uint8_t *data, uint16_t length);
  void usbMemoryW_USB1208HS(libusb_device_handle *udev, uint8_t *data, uint16_t length);
  void usbMemAddressR_USB1208HS(libusb_device_handle *udev, uint16_t address);
  void usbMemAddressW_USB1208HS(libusb_device_handle *udev, uint16_t address);
  void usbMemWriteEnable_USB1208HS(libusb_device_handle *udev);
  void usbTriggerConfig_USB1208HS(libusb_device_handle *udev, uint8_t options);
  void usbTriggerConfigR_USB1208HS(libusb_device_handle *udev, uint8_t *options);
  void usbInit_1208HS(libusb_device_handle *udev);
  void usbCounterInit_USB1208HS(libusb_device_handle *udev, uint8_t counter);
  uint32_t usbCounter_USB1208HS(libusb_device_handle *udev, uint8_t counter);
  void usbTimerControlR_USB1208HS(libusb_device_handle *udev, uint8_t *control);
  void usbTimerControlW_USB1208HS(libusb_device_handle *udev, uint8_t control);
  void usbTimerPeriodR_USB1208HS(libusb_device_handle *udev, uint32_t *period);
  void usbTimerPeriodW_USB1208HS(libusb_device_handle *udev, uint32_t period);
  void usbTimerPulseWidthR_USB1208HS(libusb_device_handle *udev, uint32_t *pulseWidth);
  void usbTimerPulseWidthW_USB1208HS(libusb_device_handle *udev, uint32_t pulseWidth);
  void usbTimerCountR_USB1208HS(libusb_device_handle *udev, uint32_t *count);
  void usbTimerCountW_USB1208HS(libusb_device_handle *udev, uint32_t count);
  void usbTimerDelayR_USB1208HS(libusb_device_handle *udev, uint32_t *delay);
  void usbTimerDelayW_USB1208HS(libusb_device_handle *udev, uint32_t delay);
  void usbTimerParamsR_USB1208HS(libusb_device_handle *udev, timerParams *params);
  void usbTimerParamsW_USB1208HS(libusb_device_handle *udev, timerParams *params);
  uint16_t usbAIn_USB1208HS(libusb_device_handle *udev, uint8_t channel);
  void usbAInConfig_USB1208HS(libusb_device_handle *udev, uint8_t mode, uint8_t range[NCHAN_1208HS]);
  void usbAInConfigR_USB1208HS(libusb_device_handle *udev, uint8_t *mode, uint8_t range[NCHAN_1208HS]);
  void usbAInScanStop_USB1208HS(libusb_device_handle *udev);
  void usbAInScanStart_USB1208HS(libusb_device_handle *udev, uint32_t count, uint32_t retrig_count, double frequency, uint8_t channels, uint8_t packet_size, uint8_t options);
  int usbAInScanRead_USB1208HS(libusb_device_handle *udev, int nScan, int nChan,  uint16_t *data, int options);
  void usbAOut_USB1208HS(libusb_device_handle *udev, uint8_t channel, double voltage, float table_AO[NCHAN_AO_1208HS][2]);
  void usbAOutR_USB1208HS(libusb_device_handle *udev, uint8_t channel, double *voltage, float table_AO[NCHAN_AO_1208HS][2]);
  void usbAOutScanStop_USB1208HS(libusb_device_handle *udev);
  void usbAOutScanClearFIFO_USB1208HS(libusb_device_handle *udev);
  void usbAOutScanStart_USB1208HS(libusb_device_handle *udev, uint32_t count, uint32_t retrig_count, double frequency, uint8_t options);
  int usbAOutScanWrite_USB1208HS(libusb_device_handle *udev, uint32_t count, uint16_t *sdata);
  void usbBuildGainTable_USB1208HS(libusb_device_handle *udev, float table[NMODE][NGAINS_1208HS][2]);
  void usbBuildGainTable_USB1208HS_4AO(libusb_device_handle *udev, float table_AO[NCHAN_AO_1208HS][2]);
  uint16_t voltsTou12_USB1208HS_AO(double volts, int channel, float table_AO[NCHAN_AO_1208HS][2]);
  double volts_USB1208HS(const uint8_t mode, const uint8_t gain, uint16_t value);

cdef extern from "mccdaq-extra.h":
  # Additional C functions to avoid manipulating ScanList object in Python world.
  ScanList *makeScanList()
  void freeScanList(ScanList *list)
  void updateScanList(ScanList *list, int item, int mode, int range, int channel)
