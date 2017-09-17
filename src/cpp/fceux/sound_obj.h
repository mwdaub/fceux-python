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

#ifndef _SOUND_H_
#define _SOUND_H_

#include "types_obj.h"

namespace fceu {

typedef struct {
       std::function<void(int)> *Fill;	/* Low quality ext sound. */

	   /* NeoFill is for sound devices that are emulated in a more
	      high-level manner(VRC7) in HQ mode.  Interestingly,
	      this device has slightly better sound quality(updated more
	      often) in lq mode than in high-quality mode.  Maybe that
     	      should be fixed. :)
	   */
       std::function<void(int32*,int)> *NeoFill;
       std::function<void(void)> *HiFill;
       std::function<void(int32)> *HiSync;

       std::function<void(void)> *RChange;
       std::function<void(void)> *Kill;
} EXPSOUND;

EXPSOUND GameExpSound;

void SetNESSoundMap(void);
void FrameSoundUpdate(void);

void FCEUSND_Power(void);
void FCEUSND_Reset(void);
void FCEUSND_SaveState(void);
void FCEUSND_LoadState(int version);

void FCEU_SoundCPUHook(int cycles);
void Write_IRQFM (uint32 A, uint8 V); //mbg merge 7/17/06 brought over from latest mmbuild

void LogDPCM(int romaddress, int dpcmsize);

uint32 SoundTimestamp(void);

typedef struct {
	uint8 Speed;
	uint8 Mode;	/* Fixed volume(1), and loop(2) */
	uint8 DecCountTo1;
	uint8 decvolume;
	int reloaddec;
} ENVUNIT;

void SetSoundVariables(void);

int GetSoundBuffer(int32 **W);
int FlushEmulateSound(void);
int32 Wave[2048+512];
int32 WaveFinal[2048+512];
int32 WaveHi[40000];
uint32 soundtsinc;

} // namespace fceu

#endif // define _SOUND_H_
