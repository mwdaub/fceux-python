#ifndef _X6502H
#define _X6502H

#include <cstring>

#include "types_obj.h"

#include "utils/general_obj.h"

#include "sound_obj.h"

#define N_FLAG  0x80
#define V_FLAG  0x40
#define U_FLAG  0x20
#define B_FLAG  0x10
#define D_FLAG  0x08
#define I_FLAG  0x04
#define Z_FLAG  0x02
#define C_FLAG  0x01

#define NTSC_CPU (dendy ? 1773447.467 : 1789772.7272727272727272)
#define PAL_CPU  1662607.125

#define FCEU_IQEXT      0x001
#define FCEU_IQEXT2     0x002
/* ... */
#define FCEU_IQRESET    0x020
#define FCEU_IQNMI2  0x040  // Delayed NMI, gets converted to *_IQNMI
#define FCEU_IQNMI  0x080
#define FCEU_IQDPCM     0x100
#define FCEU_IQFCOUNT   0x200
#define FCEU_IQTEMP     0x800

namespace FCEU {

class PPU;

class X6502 {
  public:
    void Run(int32 cycles);
    void RunDebug(int32 cycles);
    void Debug(void (*CPUHook)(X6502 *),
        uint8 (*ReadHook)(X6502 *, unsigned int),
        void (*WriteHook)(X6502 *, unsigned int, uint8));

    void Init(void);
    void Reset(void);
    void Power(void);

    void ResetGameLoaded(void) {
      MapIRQHook = NULL;
    }

    uint32 timestamp() { return timestamp_; };
    void setTimestamp(uint32 timestamp) { timestamp_ = timestamp; };
    void setSoundtimestamp(uint32 timestamp) { soundtimestamp = timestamp; };
    int32 count() { return count_; };
    uint8 DB() { return DB_; };

    void TriggerNMI(void);
    void TriggerNMI2(void);

    uint8 DMR(uint32 A);
    void DMW(uint32 A, uint8 V);

    void IRQBegin(int w);
    void IRQEnd(int w);

    #ifdef FCEUDEF_DEBUGGER
    void (*CPUHook)(struct __X6502 *);
    uint8 (*ReadHook)(struct __X6502 *, unsigned int);
    void (*WriteHook)(struct __X6502 *, unsigned int, uint8);
    #endif

  private:
    PPU* ppu;
    uint8** RAM;

    readfunc* ARead[0x10000];
    writefunc* BWrite[0x10000];

    bool overclocking;

    int32 tcount;     /* Temporary cycle counter */
    uint16 PC_;       /* I'll change this to uint32 later... */
                      /* I'll need to AND PC after increments to 0xFFFF */
                      /* when I do, though.  Perhaps an IPC() macro? */
    uint8 A_,X_,Y_,S_,P_,mooPI;
    uint8 jammed;

    int32 count_;
    uint32 IRQlow;    /* Simulated IRQ pin held low(or is it high?).
                                   And other junk hooked on for speed reasons.*/
    uint8 DB_;        /* Data bus "cache" for reads from certain areas */

    int preexec;      /* Pre-exec'ing for debug breakpoints. */

    uint32 timestamp_;
    uint32 soundtimestamp;

    void (*MapIRQHook)(int a);

    int StackAddrBackup;

    int PAL;
    int dendy;

    uint8 ZNTable[256];

    // Methods.
    inline void ADDCYC(int x) {
      tcount += x;
      count_ -= x*48;
      timestamp_ += x;
      if(!overclocking) soundtimestamp += x;
    }
    inline void PUSH(uint8 V) {
      WrRAM(0x100 + S_, V);
      S_--;
    }
    inline uint8 POP() { return RdRAM(0x100 + (++S_)); };

    //normal memory read
    inline uint8 RdMem(unsigned int A) { return(DB_=(*ARead)[A](ppu, A)); };
    inline void WrMem(unsigned int A, uint8 V) { (*BWrite)[A](ppu,A,V); };
    inline uint8 RdRAM(unsigned int A) { return(DB_=(*ARead)[A](ppu, A)); };
    inline void WrRAM(unsigned int A, uint8 V) { (*RAM)[A]=V; };
};

} // namespace FCEU

#endif // define _X6502H
