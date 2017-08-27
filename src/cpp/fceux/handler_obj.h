#ifndef _HANDLER_H_
#define _HANDLER_H_

#include "types_obj.h"

namespace FCEU {

class Handler {
  public:
    void SetReadNull(readfunc* ReadNull) { ReadNull_ = ReadNull; };
    void SetWriteNull(writefunc* WriteNull) { WriteNull_ = WriteNull; };
    void SetReadHandler(int32 pos, readfunc* func) { ARead[pos] = func; }
    void SetWriteHandler(int32 pos, writefunc* func) { BWrite[pos] = func; }

    void SetReadHandler(int32 start, int32 end, readfunc* func) {
      int32 x;

      if (!func) func = ReadNull_;
      if (RWWrap_) {
        for (x = end; x >= start; x--) {
          if (x >= 0x8000) AReadG[x - 0x8000] = func;
          else ARead[x] = func;
        }
      } else {
        for (x = end; x >= start; x--) {
          ARead[x] = func;
        }
      }
    };

    void SetWriteHandler(int32 start, int32 end, writefunc* func) {
      if (!func) func = WriteNull_;
      if (RWWrap_) {
        for (int32 x = end; x >= start; x--) {
          if (x >= 0x8000) BWriteG[x - 0x8000] = func;
          else BWrite[x] = func;
		}
      } else {
		for (int32 x = end; x >= start; x--) {
          BWrite[x] = func;
        }
      }
    };

    readfunc* GetReadHandler(int32 a) {
      if (a >= 0x8000 && RWWrap_) return AReadG[a - 0x8000];
      else return ARead[a];
    }

    writefunc* GetWriteHandler(int32 a) {
      if (RWWrap_ && a >= 0x8000) return BWriteG[a - 0x8000];
      else return BWrite[a];
    }

    int AllocGenieRW(void) {
      if (!(AReadG = (readfunc**)FCEU::malloc(0x8000 * sizeof(readfunc*))))
        return 0;
      if (!(BWriteG = (writefunc**)FCEU::malloc(0x8000 * sizeof(writefunc*))))
        return 0;
      RWWrap_ = 1;
      return 1;
    }

    void FlushGenieRW(void) {
      if (RWWrap_) {
        for (int32 x = 0; x < 0x8000; x++) {
          ARead[x + 0x8000] = AReadG[x];
          BWrite[x + 0x8000] = BWriteG[x];
        }
        FCEU::free(AReadG);
        FCEU::free(BWriteG);
        AReadG = NULL;
        BWriteG = NULL;
        RWWrap_ = 0;
      }
    }

  private:
    readfunc* ReadNull_;
    writefunc* WriteNull_;
    readfunc* ARead[0x10000];
    writefunc* BWrite[0x10000];
    readfunc** AReadG;
    writefunc** BWriteG;
    int RWWrap_;
};

} // namespace FCEU

#endif // define _HANDLER_H_
