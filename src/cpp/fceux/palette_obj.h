#ifndef _PALETTE_H_
#define _PALETTE_H_

namespace FCEU {

typedef struct {
	uint8 r,g,b;
} pal;

pal *palo;

//the default basic palette
int default_palette_selection = 0;

void FCEU_ResetPalette(void);

void FCEU_ResetPalette(void);
void FCEU_ResetMessages();
void FCEU_LoadGamePalette(void);
void FCEU_DrawNTSCControlBars(uint8 *XBuf);

void SetNESDeemph_OldHacky(uint8 d, int force);

}

#endif
