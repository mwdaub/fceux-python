#ifndef _VIDEO_H_
#define _VIDEO_H_

namespace FCEU {

uint8* XBuf=NULL; //used for current display
uint8* XBackBuf=NULL; //ppu output is stashed here before drawing happens
uint8* XDBuf=NULL; //corresponding to XBuf but with deemph bits
uint8* XDBackBuf=NULL; //corresponding to XBackBuf but with deemph bits

void FCEU_PutImage(void);
void ResetScreenshotsCounter(void);
int FCEU_InitVirtualVideo(void);
void FCEU_KillVirtualVideo(void);

}

#endif // define _VIDEO_H
