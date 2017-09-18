#ifndef _CART_H
#define _CART_H

#define MI_H 0
#define MI_V 1
#define MI_0 2
#define MI_1 3

#include "types_obj.h"

namespace fceu {

typedef struct {
	// Set by mapper/board code:
	void (*Power)(void);
	void (*Reset)(void);
	void (*Close)(void);
	uint8 *SaveGame[4];		// Pointers to memory to save/load.
	uint32 SaveGameLen[4];	// How much memory to save/load.

	// Set by iNES/UNIF loading code.
	int mirror;		// As set in the header or chunk.
					// iNES/UNIF specific.  Intended
					// to help support games like "Karnov"
					// that are not really MMC3 but are
					// set to mapper 4.
	int battery;	// Presence of an actual battery.
	int ines2;
	int submapper;	// Submappers as defined by NES 2.0
	int wram_size;
	int battery_wram_size;
	int vram_size;
	int battery_vram_size;
	uint8 MD5[16];
	uint32 CRC32;	// Should be set by the iNES/UNIF loading
					// code, used by mapper/board code, maybe
					// other code in the future.
} CartInfo;

class FCEU;

class Cart {
  friend class PPU;
  public:

    readfunc CartBR_ = [this](uint32 A) { return CartBR(A); };
    writefunc CartBW_ = [this](uint32 A, uint8 V) { CartBW(A, V); };

    void GeniePower(void);

    bool OpenGenie(void);
    void CloseGenie(void);
    void KillGenie(void);

    int GenieStage(void) { return geniestage; };

    int GetDisableBatteryLoading(void) { return disableBatteryLoading; };
    void SetDisableBatteryLoading(int val) { disableBatteryLoading = val; };

    void setntamem(uint8 *p, int ram, uint32 b);
    void setmirrorw(int a, int b, int c, int d);
    void setmirror(int t);

    void setprg2r(int r, uint32 A, uint32 V);
    void setprg2(uint32 A, uint32 V);
    void setprg4r(int r, uint32 A, uint32 V);
    void setprg4(uint32 A, uint32 V);
    void setprg8r(int r, uint32 A, uint32 V);
    void setprg8(uint32 A, uint32 V);
    void setprg16r(int r, uint32 A, uint32 V);
    void setprg16(uint32 A, uint32 V);
    void setprg32r(int r, uint32 A, uint32 V);
    void setprg32(uint32 A, uint32 V);

    void setchr1r(int r, uint32 A, uint32 V);
    void setchr2r(int r, uint32 A, uint32 V);
    void setchr4r(int r, unsigned int A, unsigned int V);
    void setchr8r(int r, uint32 V);

    void setchr1(uint32 A, uint32 V);
    void setchr2(uint32 A, uint32 V);
    void setchr4(uint32 A, uint32 V);
    void setchr8(uint32 V);

    void ResetCartMapping(void);
    void SetupCartPRGMapping(int chip, uint8 *p, uint32 size, int ram);
    void SetupCartCHRMapping(int chip, uint8 *p, uint32 size, int ram);
    void SetupCartMirroring(int m, int hard, uint8 *extra);

    void SaveGameSave(CartInfo *LocalHWInfo);
    void LoadGameSave(CartInfo *LocalHWInfo);
    void ClearGameSave(CartInfo *LocalHWInfo);

  private:
    FCEU* fceu;

    uint8 *Page[32], *VPage[8];
    uint8 **VPageR = VPage;
    uint8 *VPageG[8];
    uint8 *MMC5SPRVPage[8];
    uint8 *MMC5BGVPage[8];

    uint8 PRGIsRAM[32];  /* This page is/is not PRG RAM. */

    /* 16 are (sort of) reserved for UNIF/iNES and 16 to map other stuff. */
    uint8 CHRram[32];
    uint8 PRGram[32];

    uint8 *PRGptr[32];
    uint8 *CHRptr[32];

    uint32 PRGsize[32];
    uint32 CHRsize[32];

    uint32 PRGmask2[32];
    uint32 PRGmask4[32];
    uint32 PRGmask8[32];
    uint32 PRGmask16[32];
    uint32 PRGmask32[32];

    uint32 CHRmask1[32];
    uint32 CHRmask2[32];
    uint32 CHRmask4[32];
    uint32 CHRmask8[32];

    int geniestage = 0;
    int disableBatteryLoading = 0;

    int modcon;

    uint8 genieval[3];
    uint8 geniech[3];

    uint32 genieaddr[3];

    uint8 nothing[8192];

    int mirrorhard = 0;

    uint8 *GENIEROM = 0;

    readfunc* GenieBackup[3];

    readfunc GenieFix1_ = [this](uint32 A) { return GenieFix1(A); };
    readfunc GenieFix2_ = [this](uint32 A) { return GenieFix2(A); };
    readfunc GenieFix3_ = [this](uint32 A) { return GenieFix3(A); };

    readfunc GenieRead_ = [this](uint32 A) { return GenieRead(A); };
    writefunc GenieWrite_ = [this](uint32 A, uint8 V) { GenieWrite(A, V); };

    // Methods.
    inline void setpageptr(int s, uint32 A, uint8 *p, int ram);

    uint8 CartBR(uint32 A);
    void CartBW(uint32 A, uint8 V);
    uint8 CartBROB(uint32 A);

    uint8 GenieRead(uint32 A);
    void GenieWrite(uint32 A, uint8 V);

    uint8 GenieFix1(uint32 A);
    uint8 GenieFix2(uint32 A);
    uint8 GenieFix3(uint32 A);

    void FixGenieMap(void);
};

} // namespace fceu

#endif // define _CART_H
