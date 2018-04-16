from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Natural Language :: English',
    'Operating System :: POSIX :: Linux',
    'Programming Language :: C',
    'Programming Language :: Cython',
    'Programming Language :: Python :: 2.7',
    'Topic :: Home Automation',
    'Topic :: Scientific/Engineering'
]

usb2600_wrapper = Extension("wrapper",
		   sources = ["usb2600.pyx", "mccdaq-extra.c"],
		   library_dirs = ['.','usr/local/lib'],
		   include_dirs = ['../mcc-libhid', '.', '..', numpy.get_include(),'/usr/lib64/python/site-packages/Cython/Includes'],
		   libraries = ['usb', 'hid', 'mcchid', 'm', 'c'])


setup(
    name = 'mccdaq_linux',
    version = '1.3',
    description = "Python drivers for Measurement Computing devices (mccdaq.com) on linux",
    author = 'Guillaume Lepert',
    author_email = 'guillaume.lepert07@imperial.ac.uk',
    long_description="""Provides Python drivers for MCCDAQ data acquisition devices. 
    
    Currently the only available driver is for USB-2600 series devices,
    but hopefully that will change in the future.
    
    The package includes a "waveform" module providing simple manipulations
    of functions of times, required by the analog input/output scan functions.
    
    Example:
    
    >>> from mccdaq import usb2600, waveform  # load module
    
    >>> daq = usb2600.USB2600()     # create an instance of the driver.
    
    >>> daq.ao0(1.2)                # write 1.2V on Analog Output 0
    
    >>> daq.ai12()                    # read voltage on Analog Ouput 12
    
    >>> smooth_step = waveform.gosine(0, 1, frequency=1, rate=3000)  # define a simple wavform
    
    >>> scan = daq.AOScan((3,), smooth_step)   # create a simple AO scan on channel 3
    >>> scan.run(thread=True)                  # and run it in a background thread.
    
    Complete support for timers, counters, single-sample digital and analog input/output,
    waveform generation, synchronous analog input/output scanning, external/internal triggering;
    in a user-friendly, interactive, object-oriented format.
    """,
    cmdclass={'build_ext': build_ext},
    ext_modules = [usb2600_wrapper],
    platforms=['linux'],
    classifiers = classifiers
    
)

#A Python driver for MCCDAQ's USB2600 data acquisition devices, providing: 

    #- a Cython wrapper to Warren Jasper's original Linux MCCDAQ driver (available here_),

    #- an object-oriented interface build on top of the Cython wrapper.

    #The driver includes advanced functionalities for analog input and output scans of arbitrary length and synchronous analog input/output scanning.

    #Get in touch if anybody is interested in adding support for other MCC devices.

    #.. _here: ftp://lx10.tx.ncsu.edu/pub/Linux/drivers/USB
    #""",
