/* Provides C functions to manipulate ScanList objects in C world,
 * avoiding the headache of converting them to Python objects.
 * 
 * This file is part of the MCCDAQ Cython driver
 * Copyright Guillaume Lepert, 2014
 */

#include "usb-2600.h"

ScanList *makeScanList(void);
void freeScanList(ScanList *list);
void updateScanList(ScanList list[NCHAN_2600], int item, __u8 mode, __u8 range, __u8 channel);