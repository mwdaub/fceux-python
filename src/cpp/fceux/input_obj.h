#ifndef _INPUT_H_
#define _INPUT_H_

namespace FCEU {

unsigned int lagCounter;
char lagFlag;

void LagCounterReset();

//called from PPU on scanline events.
void InputScanlineHook(uint8 *bg, uint8 *spr, uint32 linets, int final);

void InitializeInput(void);
void FCEU_UpdateInput(void);

}

#endif // _INPUT_H_
