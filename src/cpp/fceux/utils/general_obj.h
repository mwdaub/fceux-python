/* FCE Ultra - NES/Famicom Emulator
 *
 * Copyright notice for this file:
 *  Copyright (C) 2002 Xodnizel
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 */

#ifndef _GENERAL_H_
#define _GENERAL_H_

#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../types.h"

namespace fceu {

void PrintError(char *format, ...) {
	char temp[2048];

	va_list ap;

	va_start(ap, format);
	vsnprintf(temp, sizeof(temp), format, ap);

	va_end(ap);
}
void printf(char *format, ...) {
	char temp[2048];

	va_list ap;

	va_start(ap, format);
	vsnprintf(temp, sizeof(temp), format, ap);

	va_end(ap);
}
void DispMessage(char *format, int disppos, ...);
void DispMessageOnMovie(char *format, ...);
FILE* UTF8fopen(const char *fn, const char *mode) { return(::fopen(fn,mode)); };
inline FILE* UTF8fopen(const std::string &n, const char *mode) { return UTF8fopen(n.c_str(),mode); }

///a wrapper for unzip.c
extern "C" FILE* UTF8fopen_C(const char *n, const char *m) {
	return fceu::UTF8fopen(n, m);
}

//Receives a filename (fullpath) and checks to see if that file exists
bool CheckFileExists(const char* filename) {
	//This function simply checks to see if the given filename exists
	if (!filename) return false;
    std::fstream test;
	test.open(filename, std::fstream::in);

	if (test.fail()) {
		test.close();
		return false;
	} else {
		test.close();
		return true;
	}
}

uint32 uppow2(uint32 n) {
 int x;

 for(x=31;x>=0;x--)
  if(n&(1<<x))
  {
   if((1<<x)!=n)
    return(1<<(x+1));
   break;
  }
 return n;
}

} // namespace fceu

#endif // define _GENERAL_H_
