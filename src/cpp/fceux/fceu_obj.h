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

#include "cart_obj.h"
#include "cheat_obj.h"
#include "fds_obj.h"
#include "file_obj.h"
#include "handler_obj.h"
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

namespace fceu {

class FCEU {
  friend class PPU;
  friend class FDS;
  public:
    FCEU() : handler(), input(&handler), skip_7bit_overclocking(1), AFon(1), AFoff(1),
        movieSubtitles(true), frameAdvance_Delay(FRAMEADVANCE_DELAY_DEFAULT),
        AutosaveQty(4), AutosaveFrequency(256), AutoFirePatternLength(2) {
      AutoFirePattern[0] = 1;
    };

    // Members.
    Handler handler;
    Input input;
    PPU ppu;
    X6502 x6502;
    Cart cart;
    Movie movie;
    Cheat cheat;
    Drawing drawing;
    FDS fds;
    File file;

    FCEUGI* GameInfo = NULL;
    FCEUS FSettings;

    // Methods.
    bool Initialize(void);
    void ResetGameLoaded(void);

    void Emulate(uint8 **pXBuf, int32 **SoundBuf, int32 *SoundBufSize, int skip);

    void CloseGame(void);

    void MemoryRand(uint8 *ptr, uint32 size);

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

    void SetAutoSS(bool flag) { AutoSS = flag; };

    int GetCurrentVidSystem(int *slstart, int *slend);
    void SetRenderedLines(int ntscf, int ntscl, int palf, int pall);
    void SetVidSystem(int a);

    bool IsValidUI(EFCEUI ui);

    //mbg merge 7/18/06 added
    //ideally maybe we shouldnt be using this, but i need it for quick merging
    int EmulationPaused(void) { return (EmulationPaused_ & EMULATIONPAUSED_PAUSED); };
    void SetEmulationPaused(int val) { EmulationPaused_ = val; };
    void ToggleEmulationPause(void) {
	  EmulationPaused_ = (EmulationPaused_ & EMULATIONPAUSED_PAUSED) ^ EMULATIONPAUSED_PAUSED;
	  DebuggerWasUpdated = false;
    }

  private:
    // Members.

    bool turbo;

    uint8* RAM;

    int dendy;
    int PAL;
    int pal_emulation;

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

    std::function<void(GI)> *GameInterface;
    std::function<void(int)> *GameStateRestore;

    int rapidAlternator;
    int AutoFirePattern[8];
    int AutoFirePatternLength;

	int counter;

    int RAMInitOption;

    int AutosaveCounter;

    // Methods.
    void AllocBuffers() { RAM = (uint8*)fceu::gmalloc(0x800); };
    void FreeBuffers() {
      fceu::free(RAM);
      RAM = NULL;
    };

    FCEUGI* LoadGameVirtual(const char *name, int OverwriteVidMode, bool silent);
    FCEUGI* LoadGame(const char *name, int OverwriteVidMode, bool silent);

    void Kill(void);

    void SetRegion(int region, int notify);
    //Enable or disable Game Genie option.
    void SetGameGenie(bool a) { FSettings.GameGenie = a; };
    int32 GetDesiredFPS(void);

    int EmulationFrameStepped() { return (EmulationPaused_ & EMULATIONPAUSED_FA); };
    void ClearEmulationFrameStepped() { EmulationPaused_ &= ~EMULATIONPAUSED_FA; };

    void FrameAdvanceEnd(void) { frameAdvanceRequested = false; };
    void FrameAdvance(void) {
	  frameAdvanceRequested = true;
	  frameAdvance_Delay_count = 0;
    }

    void UpdateAutosave(void);
};

} // namespace fceu

#endif
