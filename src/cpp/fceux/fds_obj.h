#ifndef _FDS_H_
#define _FDS_H_

#include "file_obj.h"
#include "git_obj.h"

namespace fceu {

typedef struct {
	int64 cycles;     // Cycles per PCM sample
	int64 count;    // Cycle counter
	int64 envcount;    // Envelope cycle counter
	uint32 b19shiftreg60;
	uint32 b24adder66;
	uint32 b24latch68;
	uint32 b17latch76;
	int32 clockcount;  // Counter to divide frequency by 8.
	uint8 b8shiftreg88;  // Modulation register.
	uint8 amplitude[2];  // Current amplitudes.
	uint8 speedo[2];
	uint8 mwcount;
	uint8 mwstart;
	uint8 mwave[0x20];      // Modulation waveform
	uint8 cwave[0x40];      // Game-defined waveform(carrier)
	uint8 SPSG[0xB];
} FDSSOUND;

class FCEU;

class FDS {
  public:
    bool isFDS = false;

    int FDSLoad(const char *name, FCEUFILE *fp);

    void FDSSoundReset(void);

    void FDSInsert(void);
    //void FCEU_FDSEject(void);
    void FDSSelect(void);

  private:
    FCEU* fceu;

    uint8 FDSRegs[6];
    int32 IRQLatch, IRQCount;
    uint8 IRQa;

    uint8 *FDSRAM = NULL;
    uint32 FDSRAMSize;
    uint8 *FDSBIOS = NULL;
    uint32 FDSBIOSsize;
    uint8 *CHRRAM = NULL;
    uint32 CHRRAMSize;

    /* Original disk data backup, to help in creating save states. */
    uint8 *diskdatao[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

    uint8 *diskdata[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

    int TotalSides; //mbg merge 7/17/06 - unsignedectomy
    uint8 DiskWritten = 0;    /* Set to 1 if disk was written to. */
    uint8 writeskip;
    int32 DiskPtr;
    int32 DiskSeekIRQ;
    uint8 SelectDisk, InDisk;

    FDSSOUND fdso;

    int ta;
    int32 FBC = 0;

    readfunc FDSRead4030_ = [this](uint32 A) { return FDSRead4030(A); };
    readfunc FDSRead4031_ = [this](uint32 A) { return FDSRead4031(A); };
    readfunc FDSRead4032_ = [this](uint32 A) { return FDSRead4032(A); };
    readfunc FDSRead4033_ = [this](uint32 A) { return FDSRead4033(A); };

    writefunc FDSWrite_ = [this](uint32 A, uint8 V) { FDSWrite(A, V); };

    readfunc FDSWaveRead_ = [this](uint32 A) { return FDSWaveRead(A); };
    writefunc FDSWaveWrite_ = [this](uint32 A, uint8 V) { FDSWaveWrite(A, V); };

    readfunc FDSSRead_ = [this](uint32 A) { return FDSSRead(A); };
    writefunc FDSSWrite_ = [this](uint32 A, uint8 V) { FDSSWrite(A, V); };

    std::function<void(int)> FDSFix_ = [this](int a) { return FDSFix(a); };
    std::function<void(int)> FDSStateRestore_ = [this](int a) { return FDSStateRestore(a); };

    std::function<void(int)> HQSync_ = [this](int a) { return HQSync(a); };
    std::function<void(void)> RenderSoundHQ_ = [this](void) { RenderSoundHQ(); };
    std::function<void(int)> FDSSound_ = [this](int a) { return FDSSound(a); };
    std::function<void(void)> FDS_ESI_ = [this](void) { FDS_ESI(); };

    std::function<void(GI)> FDSGI_ = [this](GI h) { FDSGI(h); };

    std::function<void(void)> PreSave_ = [this](void) { PreSave(); };
    std::function<void(void)> PostSave_ = [this](void) { PostSave(); };

    // Methods.
    uint8 FDSRead4030(uint32 A);
    uint8 FDSRead4031(uint32 A);
    uint8 FDSRead4032(uint32 A);
    uint8 FDSRead4033(uint32 A);

    void FDSWrite(uint32 A, uint8 V);

    void FDSWaveWrite(uint32 A, uint8 V);
    uint8 FDSWaveRead(uint32 A);

    uint8 FDSSRead(uint32 A);
    void FDSSWrite(uint32 A, uint8 V);

    void FDSInit(void);
    void FDSClose(void);
    void FDSFix(int a);
    void FDSGI(GI h);
    void FDSStateRestore(int version);
    void FDSSound(int c);
    void FDSSoundStateAdd(void);
    void RenderSound(void);
    void RenderSoundHQ(void);

    void DoEnv();

    inline void ClockRise(void);
    inline void ClockFall(void);
    inline int32 FDSDoSound(void);

    void HQSync(int32 ts);

    void FDS_ESI(void);

    void FreeFDSMemory(void);

    int SubLoad(FCEUFILE *fp);
    void PreSave(void);
    void PostSave(void);
};

} // namespace fceu

#endif // define _FDS_H_
