#ifndef _CART_H
#define _CART_H

namespace FCEU {

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

uint8 *Page[32], *VPage[8], *MMC5SPRVPage[8], *MMC5BGVPage[8];

uint8 *CHRptr[32];

int geniestage = 0;

void GeniePower(void);

bool OpenGenie(void);
void CloseGenie(void);
void KillGenie(void);

} // namespace FCEU

#endif // define _CART_H
