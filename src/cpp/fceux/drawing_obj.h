#ifndef _DRAWING_H_
#define _DRAWING_H_

#include "types_obj.h"

namespace fceu {

class FCEU;

class Drawing {
  public:
    void DrawTextLineBG(uint8 *dest);
    void DrawMessage(bool beforeMovie);
    void DrawRecordingStatus(uint8* XBuf);
    void DrawNumberRow(uint8 *XBuf, int *nstatus, int cur);
    void DrawTextTrans(uint8 *dest, uint32 width, uint8 *textmsg, uint8 fgcolor);
    void DrawTextTransWH(uint8 *dest, int width, uint8 *textmsg, uint8 fgcolor, int max_w, int max_h, int border);

  private:
    FCEU* fceu;

    char target[64][256];

    void drawstatus(uint8* XBuf, int n, int y, int xofs);

};

} // namespace fceu

#endif // define _DRAWING_H_
