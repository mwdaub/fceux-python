#ifndef _VSUNI_H_
#define _VSUNI_H_

#include "types_obj.h"
#include "git_obj.h"

namespace fceu {

SFORMAT FCEUVSUNI_STATEINFO[4];

void FCEU_VSUniPower(void);
void FCEU_VSUniCheck(uint64 md5partial, int *, uint8 *);
void FCEU_VSUniDraw(uint8 *XBuf);

void FCEU_VSUniToggleDIP(int);  /* For movies and netplay */
void FCEU_VSUniCoin(void);
void FCEU_VSUniSwap(uint8 *j0, uint8 *j1);

uint8 coinon = 0;
uint8 vsdip = 0;

} // namespace fceu

#endif // define _VSUNI_H_
