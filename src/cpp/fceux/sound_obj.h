#ifndef _SOUND_H_
#define _SOUND_H_

#include "types_obj.h"

#include "debug_obj.h"
#include "filter_obj.h"
#include "state_obj.h"
#include "wave_obj.h"

#define SOUNDTS (fceu->x6502.soundtimestamp + soundtsoffs)

namespace fceu {

typedef struct {
       std::function<void(int)> *Fill;	/* Low quality ext sound. */

	   /* NeoFill is for sound devices that are emulated in a more
	      high-level manner(VRC7) in HQ mode.  Interestingly,
	      this device has slightly better sound quality(updated more
	      often) in lq mode than in high-quality mode.  Maybe that
     	      should be fixed. :)
	   */
       std::function<void(int32*,int)> *NeoFill;
       std::function<void(void)> *HiFill;
       std::function<void(int32)> *HiSync;

       std::function<void(void)> *RChange;
       std::function<void(void)> *Kill;
} EXPSOUND;

typedef struct {
	uint8 Speed;
	uint8 Mode;	/* Fixed volume(1), and loop(2) */
	uint8 DecCountTo1;
	uint8 decvolume;
	int reloaddec;
} ENVUNIT;

class FCEU;

class Sound {
  public:
    EXPSOUND GameExpSound={0,0,0};

    int32 Wave[2048+512];
    int32 WaveFinal[2048+512];
    int32 WaveHi[40000];

    uint32 wlookup1[32];
    uint32 wlookup2[203];

    uint8 TriCount=0;
    uint8 TriMode=0;

    int32 tristep=0;

    int32 wlcount[4]={0,0,0,0};	// Wave length counters.

    // APU registers:
    uint8 PSG[0x10];			// $4000-$400F / Channels 1-4
    uint8 DMCFormat=0;			// $4010 / Play mode and frequency
    uint8 RawDALatch=0;			// $4011 / 7-bit DAC / 0xxxxxxx
    uint8 DMCAddressLatch=0;	// $4012 / Start of DMC waveform is at address $C000 + $40*$xx
    uint8 DMCSizeLatch=0;		// $4013 / Length of DMC waveform is $10*$xx + 1 bytes (128*$xx + 8 samples)
    uint8 EnabledChannels=0;	// $4015 / Sound channels enable and status
    uint8 IRQFrameMode=0;		// $4017 / Frame counter control / xx000000

    uint8 InitialRawDALatch=0; // used only for lua
    bool DMC_7bit = 0; // used to skip overclocking
    ENVUNIT EnvUnits[3];

    int32 RectDutyCount[2];
    uint8 sweepon[2];
    int32 curfreq[2];
    uint8 SweepCount[2];

    uint16 nreg=0;

    uint8 fcnt=0;
    int32 fhcnt=0;
    int32 fhinc=0;

    uint32 soundtsoffs=0;

    /* Variables exclusively for low-quality sound. */
    int32 nesincsize=0;
    uint32 soundtsinc=0;
    uint32 soundtsi=0;
    int32 sqacc[2];
    /* LQ variables segment ends. */

    int32 lengthcount[4];

    std::function<void(void)> Dummyfunc = [&](void) { return; };
    std::function<void(void)>* DoNoise = &Dummyfunc;
    std::function<void(void)>* DoTriangle = &Dummyfunc;
    std::function<void(void)>* DoPCM = &Dummyfunc;
    std::function<void(void)>* DoSQ1 = &Dummyfunc;
    std::function<void(void)>* DoSQ2 = &Dummyfunc;

    int32 DMCacc=1;
    int32 DMCPeriod=0;
    uint8 DMCBitCount=0;

    uint32 DMCAddress=0;
    int32 DMCSize=0;
    uint8 DMCShift=0;
    uint8 SIRQStat=0;

    char DMCHaveDMA=0;
    uint8 DMCDMABuf=0;
    char DMCHaveSample=0;

    uint32 ChannelBC[5];

    //savestate sync hack stuff
    int movieSyncHackOn=0,resetDMCacc=0,movieConvertOffset1,movieConvertOffset2;

    int32 inbuf=0;

    writefunc Write_IRQFM_ = [this](uint32 A, uint8 V) { Write_IRQFM(A, V); };
    writefunc Write_PSG_ = [this](uint32 A, uint8 V) { Write_PSG(A, V); };
    writefunc Write_DMCRegs_ = [this](uint32 A, uint8 V) { Write_DMCRegs(A, V); };
    writefunc StatusWrite_ = [this](uint32 A, uint8 V) { StatusWrite(A, V); };
    readfunc StatusRead_ = [this](uint32 A) { return StatusRead(A); };

    std::function<void(void)> RDoPCM_ = [this](void) { RDoPCM(); };
    std::function<void(void)> RDoSQ1_ = [this](void) { RDoSQ1(); };
    std::function<void(void)> RDoSQ2_ = [this](void) { RDoSQ2(); };
    std::function<void(void)> RDoSQLQ_ = [this](void) { RDoSQLQ(); };
    std::function<void(void)> RDoTriangle_ = [this](void) { RDoTriangle(); };
    std::function<void(void)> RDoTriangleNoisePCMLQ_ = [this](void) { RDoTriangleNoisePCMLQ(); };
    std::function<void(void)> RDoNoise_ = [this](void) { RDoNoise(); };

    SFORMAT FCEUSND_STATEINFO[40]={
        { &fhcnt, 4|FCEUSTATE_RLSB,"FHCN"},
        { &fcnt, 1, "FCNT"},
        { PSG, 0x10, "PSG"},
        { &EnabledChannels, 1, "ENCH"},
        { &IRQFrameMode, 1, "IQFM"},
        { &nreg, 2|FCEUSTATE_RLSB, "NREG"},
        { &TriMode, 1, "TRIM"},
        { &TriCount, 1, "TRIC"},
        { &EnvUnits[0].Speed, 1, "E0SP"},
        { &EnvUnits[1].Speed, 1, "E1SP"},
        { &EnvUnits[2].Speed, 1, "E2SP"},
        { &EnvUnits[0].Mode, 1, "E0MO"},
        { &EnvUnits[1].Mode, 1, "E1MO"},
        { &EnvUnits[2].Mode, 1, "E2MO"},
        { &EnvUnits[0].DecCountTo1, 1, "E0D1"},
        { &EnvUnits[1].DecCountTo1, 1, "E1D1"},
        { &EnvUnits[2].DecCountTo1, 1, "E2D1"},
        { &EnvUnits[0].decvolume, 1, "E0DV"},
        { &EnvUnits[1].decvolume, 1, "E1DV"},
        { &EnvUnits[2].decvolume, 1, "E2DV"},
        { &lengthcount[0], 4|FCEUSTATE_RLSB, "LEN0"},
        { &lengthcount[1], 4|FCEUSTATE_RLSB, "LEN1"},
        { &lengthcount[2], 4|FCEUSTATE_RLSB, "LEN2"},
        { &lengthcount[3], 4|FCEUSTATE_RLSB, "LEN3"},
        { sweepon, 2, "SWEE"},
        { &curfreq[0], 4|FCEUSTATE_RLSB,"CRF1"},
        { &curfreq[1], 4|FCEUSTATE_RLSB,"CRF2"},
        { SweepCount, 2,"SWCT"},
        { &SIRQStat, 1, "SIRQ"},
        { &DMCacc, 4|FCEUSTATE_RLSB, "5ACC"},
        { &DMCBitCount, 1, "5BIT"},
        { &DMCAddress, 4|FCEUSTATE_RLSB, "5ADD"},
        { &DMCSize, 4|FCEUSTATE_RLSB, "5SIZ"},
        { &DMCHaveDMA, 1, "5HVDM"},
        { &DMCHaveSample, 1, "5HVSP"},
        { &DMCSizeLatch, 1, "5SZL"},
        { &DMCAddressLatch, 1, "5ADL"},
        { &DMCFormat, 1, "5FMT"},
        { &RawDALatch, 1, "RWDA"},
        { 0 }
    };

    void Power(void);
    void Reset(void);
    void SaveState(void);
    void LoadState(int version);

    void SoundCPUHook(int cycles);

    void LoadDMCPeriod(uint8 V);
    void PrepDPCM();
    void LogDPCM(int romaddress, int dpcmsize);

    uint32 SoundTimestamp(void);

    void SetRate(int Rate);
    void SetLowPass(int q);
    void SetSoundQuality(int quality);
    void SetSoundVariables(void);
    void SetSoundVolume(uint32 volume);
    void SetTriangleVolume(uint32 volume);
    void SetSquare1Volume(uint32 volume);
    void SetSquare2Volume(uint32 volume);
    void SetNoiseVolume(uint32 volume);
    void SetPCMVolume(uint32 volume);

    int GetSoundBuffer(int32 **W);
    int FlushEmulateSound(void);

    int CheckFreq(uint32 cf, uint8 sr);
    void SQReload(int x, uint8 V);

    void Write_IRQFM(uint32 A, uint8 V);
    void Write_PSG(uint32 A, uint8 V);
    void Write_DMCRegs(uint32 A, uint8 V);
    void StatusWrite(uint32 A, uint8 V);
    uint8 StatusRead(uint32 A);

    void FrameSoundStuff(int V);
    void FrameSoundUpdate(void);

    inline void tester(void);
    inline void DMCDMA(void);

    void RDoPCM(void);
    inline void RDoSQ(int x);		//Int x decides if this is Square Wave 1 or 2
    void RDoSQ1(void);
    void RDoSQ2(void);
    void RDoSQLQ(void);
    void RDoTriangle(void);
    void RDoTriangleNoisePCMLQ(void);
    void RDoNoise(void);

    void SetNESSoundMap(void);

  private:
    FCEU* fceu;
};

} // namespace fceu

#endif // define _SOUND_H_
