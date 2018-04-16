# mccdaq/usb2600/.__init__.py
"""Python interface for MCCDAQ USB-26xx devices on Linux.

Two interfaces are offered:

.. toctree::
   :maxdepth: 2

   usb2600.usb2600
   usb2600.wrapper

Only the high-level interface is intended for direct use.

.. seealso::

   - `Website <http://www.mccdaq.com/usb-data-acquisition/USB-2627.aspx>`_
   
   - `User's guide <../../../../../../Datasheets/MCCDAQ/MCC_USB-2627 DAQ.pdf>`_
   
   - `Specifications <../../../../../../Datasheets/MCCDAQ/USB-2600-Series-data.pdf>`_

   - `Warren Jasper's C driver <ftp://lx10.tx.ncsu.edu/pub/Linux/drivers/USB/>`_


.. Note::
   To build the usb2600.wrapper extension, do::
     
     $ cd mccdaq/usb2600/
     $ python2.7 setup_extension.py build_ext --inplace

"""

from usb2600 import USB2600
from wrapper import USB2600_Wrapper
