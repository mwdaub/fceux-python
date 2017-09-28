#ifndef _PALETTE_H_
#define _PALETTE_H_

#include "types_obj.h"
#include "emufile_types_obj.h"

namespace fceu {

typedef struct {
	uint8 r,g,b;
} pal;

class FCEU;

class Palette {
  public:
    pal *palo;

    //the default basic palette
    int default_palette_selection = 0;

    bool force_grayscale = false;

    pal palette_game[64*8]; //custom palette for an individual game. (formerly palettei)
    pal palette_user[64*8]; //user's overridden palette (formerly palettec)
    pal palette_ntsc[64*8]; //mathematically generated NTSC palette (formerly paletten)

    bool palette_game_available; //whether palette_game is available
    bool palette_user_available; //whether palette_user is available

    //ntsc parameters:
    bool ntsccol_enable = false; //whether NTSC palette is selected
    int ntsctint = 46+10;
    int ntschue = 72;

    uint8 lastd=0;

    int controlselect=0;
    int controllength=0;

    void ResetPalette(void);
    void ResetMessages();
    void LoadGamePalette(void);
    void DrawNTSCControlBars(uint8 *XBuf);

    void SetUserPalette(uint8 *pal, int nEntries);

    void SetNESDeemph_OldHacky(uint8 d, int force);

    void CalculatePalette(void);
    void ChoosePalette(void);
    void WritePalette(void);

    void ApplyDeemphasisNTSC(int entry, u8& r, u8& g, u8& b);

    void GetNTSCTH(int *tint, int *hue);
    void SetNTSCTH(bool en, int tint, int hue);

    void NTSCDEC(void);
    void NTSCINC(void);
    void NTSCSELHUE(void);
    void NTSCSELTINT(void);

  private:
    FCEU* fceu;
};

} // namespace fceu

#endif
