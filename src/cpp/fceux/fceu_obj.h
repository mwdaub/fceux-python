#ifndef _FCEUH
#define _FCEUH

#include <fstream>
#include <sstream>
#include <string>

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <ctime>

#include "types_obj.h"
#include "git_obj.h"

#include "utils/general_obj.h"
#include "utils/memory_obj.h"

#include "cheat_obj.h"
#include "fds_obj.h"
#include "file_obj.h"
#include "ines_obj.h"
#include "movie_obj.h"
#include "nsf_obj.h"
#include "ppu_obj.h"
#include "state_obj.h"
#include "unif_obj.h"
#include "vsuni_obj.h"
#include "x6502_obj.h"


#define GAME_MEM_BLOCK_SIZE 131072

#define EMULATIONPAUSED_PAUSED 1
#define EMULATIONPAUSED_FA 2

#define FRAMEADVANCE_DELAY_DEFAULT 10

namespace FCEU {

enum GI {
	GI_RESETM2	=1,
	GI_POWER =2,
	GI_CLOSE =3,
	GI_RESETSAVE = 4
};

enum EFCEUI
{
	FCEUI_STOPAVI, FCEUI_QUICKSAVE, FCEUI_QUICKLOAD, FCEUI_SAVESTATE, FCEUI_LOADSTATE,
	FCEUI_NEXTSAVESTATE,FCEUI_PREVIOUSSAVESTATE,FCEUI_VIEWSLOTS,
	FCEUI_STOPMOVIE, FCEUI_RECORDMOVIE, FCEUI_PLAYMOVIE,
	FCEUI_OPENGAME, FCEUI_CLOSEGAME,
	FCEUI_TASEDITOR,
	FCEUI_RESET, FCEUI_POWER, FCEUI_PLAYFROMBEGINNING, FCEUI_EJECT_DISK, FCEUI_SWITCH_DISK, FCEUI_INSERT_COIN
};

class Emulator {
  public:
    Emulator() : skip_7bit_overclocking(1), AFon(1), AFoff(1),
        movieSubtitles(true), frameAdvance_Delay(FRAMEADVANCE_DELAY_DEFAULT),
        AutosaveQty(4), AutosaveFrequency(256), AutoFirePatternLength(2) {
      AutoFirePattern[0] = 1;
    };

    bool Initialize(void);
    void ResetGameLoaded(void);

    void Emulate(uint8 **pXBuf, int32 **SoundBuf, int32 *SoundBufSize, int skip);

    void CloseGame(void);

    void MemoryRand(uint8 *ptr, uint32 size);
    void SetReadHandler(int32 start, int32 end, readfunc func);
    void SetWriteHandler(int32 start, int32 end, writefunc func);
    writefunc GetWriteHandler(int32 a);
    readfunc GetReadHandler(int32 a);

    int AllocGenieRW(void);
    void FlushGenieRW(void);

    void ResetVidSys(void);

    void ResetMapping(void);
    void ResetNES(void);
    void PowerNES(void);

    void SetAutoFireOffset(int offset);
    void SetAutoFirePattern(int onframes, int offframes);
    void AutoFire(void);
    void RewindToLastAutosave(void);

    //mbg 7/23/06
    char* GetAboutString();

    uint8 ReadRomByte(uint32 i);
    void WriteRomByte(uint32 i, uint8 value);

    int TextScanlineOffset(int y) { return FSettings.FirstSLine * 256; };
    int TextScanlineOffsetFromBottom(int y) { return (FSettings.LastSLine - y) * 256; };

    void TogglePPU();

    void SetNESDeemph_OldHacky(uint8 d, int force);
    void DrawTextTrans(uint8 *dest, uint32 width, uint8 *textmsg, uint8 fgcolor);
    void PutImage(void);
    #ifdef FRAMESKIP
    void PutImageDummy(void);
    #endif

  private:
    // Members.
    PPU* ppu;
    X6502* x6502;

    FCEUS FSettings;

    int newppu;
    bool turbo;

    FCEUGI* GameInfo = NULL;

    uint8* RAM;

    int dendy;
    int PAL;
    int pal_emulation;

    readfunc ARead[0x10000];
    writefunc BWrite[0x10000];
    readfunc* AReadG;
    writefunc* BWriteG;
    int RWWrap;

    bool overclock_enabled;
    bool overclocking;
    bool skip_7bit_overclocking;
    int normalscanlines;
    int totalscanlines;
    int postrenderscanlines;
    int vblankscanlines;

    int AFon, AFoff, AutoFireOffset; //For keeping track of autofire settings
    bool justLagged;
    bool frameAdvanceLagSkip; //If this is true, frame advance will skip over lag frame (i.e. it will emulate 2 frames instead of 1)
    bool AutoSS;        //Flagged true when the first auto-savestate is made while a game is loaded, flagged false on game close
    bool movieSubtitles; //Toggle for displaying movie subtitles
    bool DebuggerWasUpdated; //To prevent the debugger from updating things without being updated.
    bool AutoResumePlay;
    char romNameWhenClosingEmulator[2048];

    uint64 timestampbase;

    int EmulationPaused_;
    bool frameAdvanceRequested;
    int frameAdvance_Delay_count;
    int frameAdvance_Delay;
    bool JustFrameAdvanced;

    int *AutosaveStatus; //is it safe to load Auto-savestate
    int AutosaveIndex; //which Auto-savestate we're on
    int AutosaveQty; // Number of Autosaves to store
    int AutosaveFrequency; // Number of frames between autosaves

    // Flag that indicates whether the Auto-save option is enabled or not
    int EnableAutosave;

    void (*GameInterface)(GI h);
    void (*GameStateRestore)(int version);

    int rapidAlternator;
    int AutoFirePattern[8];
    int AutoFirePatternLength;

	int counter;

    int RAMInitOption;

    int AutosaveCounter;

    // Methods.
    void AllocBuffers() { RAM = (uint8*)FCEU::gmalloc(0x800); };
    void FreeBuffers() {
      FCEU::free(RAM);
      RAM = NULL;
    };

    FCEUGI* LoadGameVirtual(const char *name, int OverwriteVidMode, bool silent);
    FCEUGI* LoadGame(const char *name, int OverwriteVidMode, bool silent);

    void Kill(void);

    void SetRenderedLines(int ntscf, int ntscl, int palf, int pall);
    void SetVidSystem(int a);
    int GetCurrentVidSystem(int *slstart, int *slend);

    void SetRegion(int region, int notify);
    //Enable or disable Game Genie option.
    void SetGameGenie(bool a) { FSettings.GameGenie = a; };
    int32 GetDesiredFPS(void);

    int EmulationPaused(void) { return (EmulationPaused_ & EMULATIONPAUSED_PAUSED); };
    int EmulationFrameStepped() { return (EmulationPaused_ & EMULATIONPAUSED_FA); };
    void ClearEmulationFrameStepped() { EmulationPaused_ &= ~EMULATIONPAUSED_FA; };

    //mbg merge 7/18/06 added
    //ideally maybe we shouldnt be using this, but i need it for quick merging
    void SetEmulationPaused(int val) { EmulationPaused_ = val; };
    void ToggleEmulationPause(void) {
	  EmulationPaused_ = (EmulationPaused_ & EMULATIONPAUSED_PAUSED) ^ EMULATIONPAUSED_PAUSED;
	  DebuggerWasUpdated = false;
    }

    void FrameAdvanceEnd(void) { frameAdvanceRequested = false; };
    void FrameAdvance(void) {
	  frameAdvanceRequested = true;
	  frameAdvance_Delay_count = 0;
    }

    void UpdateAutosave(void);

    bool IsValidUI(EFCEUI ui);
};

}

#endif