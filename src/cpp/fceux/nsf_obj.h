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

#ifndef NSF_H
#define NSF_H

#include "types_obj.h"
#include "git_obj.h"

#include "file_obj.h"
#include "state_obj.h"

namespace fceu {

typedef struct {
                char ID[5]; /*NESM^Z*/
                uint8 Version;
                uint8 TotalSongs;
                uint8 StartingSong;
                uint8 LoadAddressLow;
                uint8 LoadAddressHigh;
                uint8 InitAddressLow;
                uint8 InitAddressHigh;
                uint8 PlayAddressLow;
                uint8 PlayAddressHigh;
                uint8 SongName[32];
                uint8 Artist[32];
                uint8 Copyright[32];
                uint8 NTSCspeed[2];        // Unused
                uint8 BankSwitch[8];
                uint8 PALspeed[2];         // Unused
                uint8 VideoSystem;
                uint8 SoundChip;
                uint8 Expansion[4];
                uint8 reserve[8];
        } NSF_HEADER;

class FCEU;

class NSF {
  public:
    FCEU* fceu;

    NSF_HEADER NSFHeader; //mbg merge 6/29/06
    uint8 *NSFDATA = 0;
    int NSFMaxBank;

    void NSF_init(void);
    void DrawNSF(uint8 *XBuf);
    void NSFDealloc(void);
    void NSFDodo(void);
    void DoNSFFrame(void);

    int NSFLoad(const char *name, FCEUFILE *fp);

    uint8 SongReload;
    int32 CurrentSong;

    int vismode=1; //we cant consider this state, because the UI may be controlling it and wouldnt know we loadstated it

    uint8 doreset=0; //state
    uint8 NSFNMIFlags; //state

    int32 NSFSize; //configuration
    uint8 BSon; //configuration
    uint8 BankCounter; //configuration

    uint16 PlayAddr; //configuration
    uint16 InitAddr; //configuration
    uint16 LoadAddr; //configuration

    uint8 *ExWRAM=0;

    //zero 17-apr-2013 - added
    SFORMAT StateRegs[5] = {
	    {&SongReload, 1, "SREL"},
	    {&CurrentSong, 4 | FCEUSTATE_RLSB, "CURS"},
	    {&doreset, 1, "DORE"},
	    {&NSFNMIFlags, 1, "NMIF"},
	    { 0 }
    };

    int special=0;

	uint8 last=0;


    void NSFMMC5_Close(void);
    void NSFGI(GI h);

    std::function<void(GI)> NSFGI_ = [this](GI h) { NSFGI(h); };

    inline void BANKSET(uint32 A, uint32 bank);

    uint8 NSFROMRead(uint32 A);
    uint8 NSFVectorRead(uint32 A);
    void NSF_write(uint32 A, uint8 V);
    uint8 NSF_read(uint32 A);

    readfunc NSFROMRead_ = [this](uint32 A) { return NSFROMRead(A); };
    readfunc NSFVectorRead_ = [this](uint32 A) { return NSFVectorRead(A); };
    readfunc NSF_read_ = [this](uint32 A) { return NSF_read(A); };
    writefunc NSF_write_ = [this](uint32 A, uint8 V) { return NSF_write(A, V); };

    void NSFSetVis(int mode);
    int NSFChange(int amount);
    int NSFGetInfo(uint8 *name, uint8 *artist, uint8 *copyright, int maxlen);
};

void NSFVRC6_Init(void);
void NSFVRC7_Init(void);
void NSFMMC5_Init(void);
void NSFN106_Init(void);
void NSFAY_Init(void);

} // namespace fceu

#endif // define NSF_H
