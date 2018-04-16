/* Provides C functions to manipulate ScanList objects in C world,
 * avoiding the headache of converting them to Python objects.
 * 
 * This file is part of the MCCDAQ Cython driver
 * Copyright Guillaume Lepert, 2014
 */

#include <asm/types.h>
#include "usb-2600.h"
#include "mccdaq-extra.h"

ScanList *makeScanList(void) {
  ScanList *list = malloc(NCHAN_2600*sizeof(ScanList));
  return list;
}

void freeScanList(ScanList *list) {
  free(list);
}

void updateScanList(ScanList list[NCHAN_2600], int item, __u8 mode, __u8 range, __u8 channel) {
  list[item].range = range;
  list[item].mode = mode;
  list[item].channel = channel;
}