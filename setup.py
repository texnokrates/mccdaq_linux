from setuptools import setup, Extension, find_packages
from Cython.Distutils import build_ext
import numpy
import sys

sys.path.append('/media/labdata/Brillouin/Python/utilities')

print sys.path

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

usb2600_wrapper = Extension("mccdaq.usb2600.wrapper",
		   sources = ["mccdaq/usb2600/usb2600.pyx", "mccdaq/usb2600/mccdaq-extra.c"],
		   library_dirs = ['.','usr/local/lib'],
		   include_dirs = ['mccdaq/usb2600', '.', '..', numpy.get_include(),'/usr/lib64/python/site-packages/Cython/Includes', 'mcc-libhid', '/opt/mcc-libhid'],
		   libraries = ['usb', 'hid', 'mcchid', 'm', 'c'])

print find_packages()

setup(
    name = 'mccdaq_linux',
    version = '1.4.8',
    description = "Python drivers for Measurement Computing devices (mccdaq.com) on linux",
    author = 'Guillaume Lepert',
    author_email = 'guillaume.lepert07@imperial.ac.uk',
    long_description="""Python drivers for data acquisition hardware from Measurement Computing (http://mccdaq.com/).

    Currently only USB26xx devices are supported, but we could add more.
    
    Complete support for timers, counters, single-sample digital and analog input/output,
    waveform generation, synchronous analog input/output scanning, external/internal triggering;
    in a user-friendly, interactive, object-oriented format.
    
    Installation notice
    -------------------
    
    `mccdaq_linux` expects:
    
      1) that Warren Jaspers' C drivers are already installed (see ftp://lx10.tx.ncsu.edu/pub/Linux/drivers/USB/)
      
      2) that said driver's source files be under /opt/mcc-libhid.
    
    This being the case, use pip as usual::
    
        $ pip2.7 install mccdaq_linux
      
    """,
    packages=['mccdaq', 'mccdaq.usb2600', 'mccdaq.utilities'],
    #package_dir={'mccdaq.utilities': '../../utilities'},
    cmdclass={'build_ext': build_ext},
    ext_modules = [usb2600_wrapper],
    install_requires = ['numpy', 'matplotlib'],
    platforms=['linux'],
    classifiers = classifiers   
)
