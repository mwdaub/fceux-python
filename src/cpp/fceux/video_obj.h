#ifndef _VIDEO_H_
#define _VIDEO_H_

#include "types_obj.h"

#include "utils/endian_obj.h"

namespace fceu {

struct GUIMESSAGE
{
	//countdown for gui messages
	int howlong;

	//the current gui message
	char errmsg[110];

	//indicates that the movie should be drawn even on top of movies
	bool isMovieMessage;

	//in case of multiple lines, allow one to move the message
	int linesFromBottom;

};

uint8* XBuf=NULL; //used for current display
uint8* XBackBuf=NULL; //ppu output is stashed here before drawing happens
uint8* XDBuf=NULL; //corresponding to XBuf but with deemph bits
uint8* XDBackBuf=NULL; //corresponding to XBackBuf but with deemph bits
int ClipSidesOffset=0;	//Used to move displayed messages when Clips left and right sides is checked
u8 *xbsave=NULL;

GUIMESSAGE guiMessage;
GUIMESSAGE subtitleMessage;

void FCEU_PutImage(void);
void ResetScreenshotsCounter(void);
int FCEU_InitVirtualVideo(void);
void FCEU_KillVirtualVideo(void);

} // namespace fceu

#endif // define _VIDEO_H
