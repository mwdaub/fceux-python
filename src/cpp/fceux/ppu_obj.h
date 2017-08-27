#ifndef _PPUH
#define _PPUH

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <functional>

#include "types_obj.h"
#include "git_obj.h"

#include "utils/general_obj.h"
#include "utils/memory_obj.h"

#include "cart_obj.h"
#include "handler_obj.h"
#include "input_obj.h"
#include "nsf_obj.h"
#include "palette_obj.h"
#include "state_obj.h"
#include "video_obj.h"
#include "x6502_obj.h"

#ifdef _MSC_VER
#define FASTCALL __fastcall
#else
#define FASTCALL
#endif

#define V_FLIP  0x80
#define H_FLIP  0x40
#define SP_BACK 0x20

namespace FCEU {

template<typename T, int BITS>
struct BITREVLUT {
	T* lut;
	BITREVLUT() {
		int bits = BITS;
		int n = 1 << BITS;
		lut = new T[n];

		int m = 1;
		int a = n >> 1;
		int j = 2;

		lut[0] = 0;
		lut[1] = a;

		while (--bits) {
			m <<= 1;
			a >>= 1;
			for (int i = 0; i < m; i++)
				lut[j++] = lut[i] + a;
		}
	}

	T operator[](int index) {
		return lut[index];
	}
};

struct PPUSTATUS {
	int32 sl;
	int32 cycle, end_cycle;
};

struct SPRITE_READ {
	int32 num;
	int32 count;
	int32 fetch;
	int32 found;
	int32 found_pos[8];
	int32 ret;
	int32 last;
	int32 mode;

	void reset() {
		num = count = fetch = found = ret = last = mode = 0;
		found_pos[0] = found_pos[1] = found_pos[2] = found_pos[3] = 0;
		found_pos[4] = found_pos[5] = found_pos[6] = found_pos[7] = 0;
	}

	void start_scanline() {
		num = 1;
		found = 0;
		fetch = 1;
		count = 0;
		last = 64;
		mode = 0;
		found_pos[0] = found_pos[1] = found_pos[2] = found_pos[3] = 0;
		found_pos[4] = found_pos[5] = found_pos[6] = found_pos[7] = 0;
	}
};

//uses the internal counters concept at http://nesdev.icequake.net/PPU%20addressing.txt
struct PPUREGS {
	//normal clocked regs. as the game can interfere with these at any time, they need to be savestated
	uint32 fv;	//3
	uint32 v;	//1
	uint32 h;	//1
	uint32 vt;	//5
	uint32 ht;	//5

	//temp unlatched regs (need savestating, can be written to at any time)
	uint32 _fv, _v, _h, _vt, _ht;

	//other regs that need savestating
	uint32 fh;	//3 (horz scroll)
	uint32 s;	//1 ($2000 bit 4: "Background pattern table address (0: $0000; 1: $1000)")

	//other regs that don't need saving
	uint32 par;	//8 (sort of a hack, just stored in here, but not managed by this system)

	//cached state data. these are always reset at the beginning of a frame and don't need saving
	//but just to be safe, we're gonna save it
	PPUSTATUS status;

	void reset() {
		fv = v = h = vt = ht = 0;
		fh = par = s = 0;
		_fv = _v = _h = _vt = _ht = 0;
		status.cycle = 0;
		status.end_cycle = 341;
		status.sl = 241;
	}

	void install_latches() {
		fv = _fv;
		v = _v;
		h = _h;
		vt = _vt;
		ht = _ht;
	}

	void install_h_latches() {
		ht = _ht;
		h = _h;
	}

	void clear_latches() {
		_fv = _v = _h = _vt = _ht = 0;
		fh = 0;
	}

	void increment_hsc() {
		//The first one, the horizontal scroll counter, consists of 6 bits, and is
		//made up by daisy-chaining the HT counter to the H counter. The HT counter is
		//then clocked every 8 pixel dot clocks (or every 8/3 CPU clock cycles).
		ht++;
		h += (ht >> 5);
		ht &= 31;
		h &= 1;
	}

	void increment_vs() {
		fv++;
		int fv_overflow = (fv >> 3);
		vt += fv_overflow;
		vt &= 31;	//fixed tecmo super bowl
		if (vt == 30 && fv_overflow == 1) {	//caution here (only do it at the exact instant of overflow) fixes p'radikus conflict
			v++;
			vt = 0;
		}
		fv &= 7;
		v &= 1;
	}

	uint32 get_ntread() {
		return 0x2000 | (v << 0xB) | (h << 0xA) | (vt << 5) | ht;
	}

	uint32 get_2007access() {
		return ((fv & 3) << 0xC) | (v << 0xB) | (h << 0xA) | (vt << 5) | ht;
	}

	//The PPU has an internal 4-position, 2-bit shifter, which it uses for
	//obtaining the 2-bit palette select data during an attribute table byte
	//fetch. To represent how this data is shifted in the diagram, letters a..c
	//are used in the diagram to represent the right-shift position amount to
	//apply to the data read from the attribute data (a is always 0). This is why
	//you only see bits 0 and 1 used off the read attribute data in the diagram.
	uint32 get_atread() {
		return 0x2000 | (v << 0xB) | (h << 0xA) | 0x3C0 | ((vt & 0x1C) << 1) | ((ht & 0x1C) >> 2);
	}

	//address line 3 relates to the pattern table fetch occuring (the PPU always makes them in pairs).
	uint32 get_ptread() {
		return (s << 0xC) | (par << 0x4) | fv;
	}

	void increment2007(bool rendering, bool by32) {

		if (rendering)
		{
			//don't do this:
			//if (by32) increment_vs();
			//else increment_hsc();
			//do this instead:
			increment_vs();  //yes, even if we're moving by 32
			return;
		}

		//If the VRAM address increment bit (2000.2) is clear (inc. amt. = 1), all the
		//scroll counters are daisy-chained (in the order of HT, VT, H, V, FV) so that
		//the carry out of each counter controls the next counter's clock rate. The
		//result is that all 5 counters function as a single 15-bit one. Any access to
		//2007 clocks the HT counter here.
		//
		//If the VRAM address increment bit is set (inc. amt. = 32), the only
		//difference is that the HT counter is no longer being clocked, and the VT
		//counter is now being clocked by access to 2007.
		if (by32) {
			vt++;
		} else {
			ht++;
			vt += (ht >> 5) & 1;
		}
		h += (vt >> 5);
		v += (h >> 1);
		fv += (v >> 1);
		ht &= 31;
		vt &= 31;
		h &= 1;
		v &= 1;
		fv &= 7;
	}

	void debug_log()
	{
        FCEU::printf("ppur: fv(%d), v(%d), h(%d), vt(%d), ht(%d)\n",fv,v,h,vt,ht);
        FCEU::printf("      _fv(%d), _v(%d), _h(%d), _vt(%d), _ht(%d)\n",_fv,_v,_h,_vt,_ht);
        FCEU::printf("      fh(%d), s(%d), par(%d)\n",fh,s,par);
        FCEU::printf("      .status cycle(%d), end_cycle(%d), sl(%d)\n",status.cycle,status.end_cycle,status.sl);
	}
};

typedef struct {
	uint8 y, no, atr, x;
} SPR;

typedef struct {
	uint8 ca[2], atr, x;
} SPRB;

enum PPUPHASE {
	PPUPHASE_VBL, PPUPHASE_BG, PPUPHASE_OBJ
};

typedef struct {
  int debug_loggingCD;
  unsigned char* cdloggervdata;
  int cdloggerVideoDataSize;
} LOGGER;

class PPU;

typedef uint8 (FASTCALL *ppureadfunc)(PPU* ppu, uint32 A);
typedef void (*ppuwritefunc)(PPU* ppu, uint32 A, uint8 V);

class PPU {
  public:
    // Constructor.
    PPU(uint8 PAL, LOGGER logger) : PAL(PAL), gNoBGFillColor(0xFF), new_ppu_reset(false), idleSynch(1),
        ppudead(1), kook(0), fceuindbg(0), maxsprites(8), rendercount(-1), vromreadcount(-1),
        undefinedvromcount(-1), LogAddress(-1), rendersprites(true), renderbg(true),
        bgdata(this), logger(logger) {
      SFORMAT stateinfo[14] = {
        { NTARAM, 0x800, "NTAR" },
        { PALRAM, 0x20, "PRAM" },
        { SPRAM, 0x100, "SPRA" },
        { data, 0x4, "PPUR" },
        { &kook, 1, "KOOK" },
        { &ppudead, 1, "DEAD" },
        { &PPUSPL, 1, "PSPL" },
        { &XOffset, 1, "XOFF" },
        { &vtoggle, 1, "VTGL" },
        { &RefreshAddrT, 2 | FCEUSTATE_RLSB, "RADD" },
        { &TempAddrT, 2 | FCEUSTATE_RLSB, "TADD" },
        { &VRAMBuffer, 1, "VBUF" },
        { &PPUGenLatch, 1, "PGEN" },
        { 0 }
      };
      memcpy(FCEUPPU_STATEINFO, stateinfo, 14*sizeof(SFORMAT));

      SFORMAT newstateinfo[32]= {
	    { &idleSynch, 1, "IDLS" },
	    { &spr_read.num, 4 | FCEUSTATE_RLSB, "SR_0" },
	    { &spr_read.count, 4 | FCEUSTATE_RLSB, "SR_1" },
	    { &spr_read.fetch, 4 | FCEUSTATE_RLSB, "SR_2" },
	    { &spr_read.found, 4 | FCEUSTATE_RLSB, "SR_3" },
	    { &spr_read.found_pos[0], 4 | FCEUSTATE_RLSB, "SRx0" },
	    { &spr_read.found_pos[0], 4 | FCEUSTATE_RLSB, "SRx1" },
	    { &spr_read.found_pos[0], 4 | FCEUSTATE_RLSB, "SRx2" },
	    { &spr_read.found_pos[0], 4 | FCEUSTATE_RLSB, "SRx3" },
	    { &spr_read.found_pos[0], 4 | FCEUSTATE_RLSB, "SRx4" },
	    { &spr_read.found_pos[0], 4 | FCEUSTATE_RLSB, "SRx5" },
	    { &spr_read.found_pos[0], 4 | FCEUSTATE_RLSB, "SRx6" },
	    { &spr_read.found_pos[0], 4 | FCEUSTATE_RLSB, "SRx7" },
	    { &spr_read.ret, 4 | FCEUSTATE_RLSB, "SR_4" },
	    { &spr_read.last, 4 | FCEUSTATE_RLSB, "SR_5" },
	    { &spr_read.mode, 4 | FCEUSTATE_RLSB, "SR_6" },
	    { &ppur.fv, 4 | FCEUSTATE_RLSB, "PFVx" },
	    { &ppur.v, 4 | FCEUSTATE_RLSB, "PVxx" },
	    { &ppur.h, 4 | FCEUSTATE_RLSB, "PHxx" },
	    { &ppur.vt, 4 | FCEUSTATE_RLSB, "PVTx" },
	    { &ppur.ht, 4 | FCEUSTATE_RLSB, "PHTx" },
	    { &ppur._fv, 4 | FCEUSTATE_RLSB, "P_FV" },
	    { &ppur._v, 4 | FCEUSTATE_RLSB, "P_Vx" },
	    { &ppur._h, 4 | FCEUSTATE_RLSB, "P_Hx" },
	    { &ppur._vt, 4 | FCEUSTATE_RLSB, "P_VT" },
	    { &ppur._ht, 4 | FCEUSTATE_RLSB, "P_HT" },
	    { &ppur.fh, 4 | FCEUSTATE_RLSB, "PFHx" },
	    { &ppur.s, 4 | FCEUSTATE_RLSB, "PSxx" },
	    { &ppur.status.sl, 4 | FCEUSTATE_RLSB, "PST0" },
	    { &ppur.status.cycle, 4 | FCEUSTATE_RLSB, "PST1" },
	    { &ppur.status.end_cycle, 4 | FCEUSTATE_RLSB, "PST2" },
	    { 0 }
      };
      memcpy(FCEU_NEWPPU_STATEINFO, newstateinfo, 32*sizeof(SFORMAT));
    }

    // Members.
    bool* DMC_7bit;
    bool* paldeemphswap;

    readfunc* FFCEUX_PPURead;
    writefunc* FFCEUX_PPUWrite;

    void (*GameHBIRQHook)(void), (*GameHBIRQHook2)(void);
    void (*PPU_hook)(uint32 A);

    // Methods.
    //Initializes the PPU
    void Init(void) { makeppulut(); };
    void Reset(void);
    void Power(void);
    int Loop(int skip);
    int NewLoop(int skip);

    void ResetGameLoaded(void) {
	  PPU_hook = NULL;
	  GameHBIRQHook = NULL;
	  FFCEUX_PPURead = NULL;
	  FFCEUX_PPUWrite = NULL;
	  MMC5Hack = 0;
	  PEC586Hack = 0;
    }

    void LineUpdate();
    void SetVideoSystem(int w);

    int newppu_get_scanline() { return ppur.status.sl; };
    int newppu_get_dot() { return ppur.status.cycle; };
    void newppu_hacky_emergency_reset() {
	    if(ppur.status.end_cycle == 0) ppur.reset();
    }

    void SaveState(void);
    void LoadState(int version);
    uint32 PeekAddress();
    uint8* GetCHR(uint32 vadr, uint32 refreshaddr);
    int GetAttr(int ntnum, int xt, int yt);
    void getScroll(int &xpos, int &ypos);

    void ResetHooks() { FFCEUX_PPURead = &Read_Default_; }

    inline uint8 VBlankON() { return data[0] & 0x80; };	//Generate VBlank NMI
    inline uint8 Sprite16() { return data[0] & 0x20; };	//Sprites 8x16/8x8
    inline uint8 BGAdrHI() { return data[0] & 0x10; };	//BG pattern adr $0000/$1000
    inline uint8 SpAdrHI() { return data[0] & 0x08; };	//Sprite pattern adr $0000/$1000
    inline uint8 INC32() { return data[0] & 0x04; };	//auto increment 1/32

    inline uint8 SpriteON() { return data[1] & 0x10; };	//Show Sprite
    inline uint8 ScreenON() { return data[1] & 0x08; };	//Show screen
    inline uint8 PPUON() { return data[1] & 0x18; };	//PPU should operate
    inline uint8 GRAYSCALE() { return data[1] & 0x01; };	//Grayscale (AND palette entries with 0x30)

    inline uint8 SpriteLeft8() { return data[1] & 0x04; };
    inline uint8 BGLeft8() { return data[1] & 0x02;} ;

    inline uint8 Status() { return data[2]; };
    inline void updateStatus(uint8 s) { data[2] = s; };

    inline uint8 READPAL(int ofs) { return PALRAM[ofs] & (GRAYSCALE() ? 0x30 : 0xFF); };
    inline uint8 READUPAL(int ofs) { return UPALRAM[ofs] & (GRAYSCALE() ? 0x30 : 0xFF); };

    inline uint8* MMC5SPRVRAMADR(int V) { return &MMC5SPRVPage[V >> 10][V]; };
    inline uint8* VRAMADR(int V) { return &VPage[V >> 10][V]; };

    inline uint8 Read(uint32 A) { return (*FFCEUX_PPURead)(A); };
    inline void Write(uint32 A, uint8 V) { FFCEUX_PPUWrite ? (*FFCEUX_PPUWrite)(A, V) : Write_Default(A, V); };

    inline int GETLASTPIXEL() { return PAL ? ((x6502->timestamp() * 48 - linestartts) / 15) : ((x6502->timestamp() * 48 - linestartts) >> 4); };

    readfunc ANull_ = [this](uint32 A) { return(this->x6502->DB()); };
    writefunc BNull_ = [this](uint32 A, uint8 V) {};
    writefunc BRAML_ = [this](uint32 A, uint8 V) { (*(this->RAM))[A] = V; };
    writefunc BRAMH_ = [this](uint32 A, uint8 V) { (*(this->RAM))[A & 0x7FF] = V; };
    readfunc  ARAML_ = [this](uint32 A) { return (*(this->RAM))[A]; };
    readfunc ARAMH_ = [this](uint32 A) { return (*(this->RAM))[A & 0x7FF]; };

  private:
    // Members.
    static BITREVLUT<uint8, 8> bitrevlut;

    Handler* handler;

    X6502* x6502;
    uint8** RAM;

    bool overclock_enabled;
    bool overclocking;
    bool skip_7bit_overclocking;
    int normalscanlines;
    int totalscanlines;
    int postrenderscanlines;
    int vblankscanlines;

    FCEUGI* GameInfo;

    FCEUS* FSettings;

    int dendy;
    uint8 PAL;

    //mbg 6/23/08
    //make the no-bg fill color configurable
    //0xFF shall indicate to use palette[0]
    uint8 gNoBGFillColor;

    uint32 ppulut1[256];
    uint32 ppulut2[256];
    uint32 ppulut3[128];

    bool new_ppu_reset;

    //doesn't need to be savestated as it is just a reflection of the current position in the ppu loop
    PPUPHASE ppuphase;

    //this needs to be savestated since a game may be trying to read from this across vblanks
    SPRITE_READ spr_read;

    //definitely needs to be savestated
    uint8 idleSynch;

    PPUREGS ppur;

    int ppudead;
    int kook;
    int fceuindbg;

    int MMC5Hack, PEC586Hack;
    uint32 MMC5HackVROMMask;
    uint8 *MMC5HackExNTARAMPtr;
    uint8 *MMC5HackVROMPTR;
    uint8 MMC5HackCHRMode;
    uint8 MMC5HackSPMode;
    uint8 MMC50x5130;
    uint8 MMC5HackSPScroll;
    uint8 MMC5HackSPPage;

    uint8 VRAMBuffer, PPUGenLatch;
    uint8 *vnapage[4];
    uint8 PPUNTARAM;
    uint8 PPUCHRRAM;

    //Color deemphasis emulation.  Joy...
    uint8 deemp;
    int deempcnt[8];

    uint8 vtoggle;
    uint8 XOffset;
    uint8 SpriteDMA; // $4014 / Writing $xx copies 256 bytes by reading from $xx00-$xxFF and writing to $2004 (OAM data)

    uint32 TempAddr, RefreshAddr, DummyRead, NTRefreshAddr;

    int maxsprites;

    //scanline is equal to the current visible scanline we're on.
    int scanline;
    int g_rasterpos;
    uint32 scanlines_per_frame;

    uint8 data[4]; // Rename from original PPU so it doesn't conflict with class name.
    uint8 PPUSPL;
    uint8 NTARAM[0x800], PALRAM[0x20], SPRAM[0x100], SPRBUF[0x100];
    // for 0x4/0x8/0xC addresses in palette, the ones in 0x20 are 0 to not break fceu rendering.
    uint8 UPALRAM[0x03];

    uint8* (*MMC5BGVRAMADR)(uint32 A);

    volatile int rendercount, vromreadcount, undefinedvromcount, LogAddress;
    unsigned char *cdloggervdata;
    unsigned int cdloggerVideoDataSize;

    //whether to use the new ppu
    int newppu;

    uint8 *Pline, *Plinef;
    int firsttile;
    int linestartts;	//no longer static so the debugger can see it
    int tofix;

    uint8 sprlinebuf[256 + 8];

    bool rendersprites, renderbg;

    int32 sphitx;
    uint8 sphitdata;

    //spork the world.  Any sprites on this line? Then this will be set to 1.
    //Needed for zapper emulation and *gasp* sprite emulation.
    int spork;

    uint8 numsprites, SpriteBlurp;

    uint16 TempAddrT, RefreshAddrT;

    SFORMAT FCEUPPU_STATEINFO[14];
    SFORMAT FCEU_NEWPPU_STATEINFO[32];

    //---------------------
    int pputime;
    int totpputime;
    static const int kLineTime = 341;
    static const int kFetchTime = 2;

    //todo - consider making this a 3 or 4 slot fifo to keep from touching so much memory
    struct BGData {
      BGData(PPU* ppu) {
        for (int i = 0; i < 34; i++) {
          main[i].ppu = ppu;
        }
      }

      struct Record {
        uint8 nt, pecnt, at, pt[2];
        PPU* ppu;
        inline void Read() {
          ppu->Read(nt, pecnt, at, pt);
        }
      };

      Record main[34];	//one at the end is junk, it can never be rendered
    } bgdata;

    int framectr;

	uint32 pshift[2];
	uint32 atlatch;
	int norecurse;	// Yeah, recursion would be bad.
				  	// PPU_hook() functions can call
					// mirroring/chr bank switching functions,
					// which call FCEUPPU_LineUpdate, which call this
					// function.

    uint8 oams[2][64][8]; //[7] turned to [8] for faster indexing
    int oamcounts[2];
    int oamslot;
    int oamcount;

    // Methods.
    void FetchSpriteData(void);
    void RefreshLine(int lastpixel);
    void RefreshSprites(void);
    void CopySprites(uint8 *target);

    void Fixit1(void);
    INLINE void Fixit2(void);

    void makeppulut(void);

    uint8 FASTCALL Read_Default(uint32 A);
    void Write_Default(uint32 A, uint8 V);
    readfunc Read_Default_ = [this](uint32 A) { return this->Read_Default(A); };

    int GetCHRAddress(int A);

    uint8 A2002(uint32 A);
    uint8 A2004(uint32 A);
    uint8 A200x(uint32 A);
    uint8 A2007(uint32 A);
    void B2000(uint32 A, uint8 V);
    void B2001(uint32 A, uint8 V);
    void B2002(uint32 A, uint8 V);
    void B2003(uint32 A, uint8 V);
    void B2004(uint32 A, uint8 V);
    void B2005(uint32 A, uint8 V);
    void B2006(uint32 A, uint8 V);
    void B2007(uint32 A, uint8 V);
    void B4014(uint32 A, uint8 V);

    readfunc A2002_ = [this](uint32 A) { return this->A2002(A); };
    readfunc A2004_ = [this](uint32 A) { return this->A2004(A); };
    readfunc A200x_ = [this](uint32 A) { return this->A200x(A); };
    readfunc A2007_ = [this](uint32 A) { return this->A2007(A); };
    writefunc B2000_ = [this](uint32 A, uint8 V) { this->B2000(A, V); };
    writefunc B2001_ = [this](uint32 A, uint8 V) { this->B2001(A, V); };
    writefunc B2002_ = [this](uint32 A, uint8 V) { this->B2002(A, V); };
    writefunc B2003_ = [this](uint32 A, uint8 V) { this->B2003(A, V); };
    writefunc B2004_ = [this](uint32 A, uint8 V) { this->B2004(A, V); };
    writefunc B2005_ = [this](uint32 A, uint8 V) { this->B2005(A, V); };
    writefunc B2006_ = [this](uint32 A, uint8 V) { this->B2006(A, V); };
    writefunc B2007_ = [this](uint32 A, uint8 V) { this->B2007(A, V); };
    writefunc B4014_ = [this](uint32 A, uint8 V) { this->B4014(A, V); };

    void ResetRL(uint8 *target);

    void SetRenderPlanes(bool sprites, bool bg);
    void GetRenderPlanes(bool& sprites, bool& bg);

    void CheckSpriteHit(int p);

    void EndRL(void);

    void DoLine(void);

    void DisableSpriteLimitation(int a) {
	    maxsprites = a ? 64 : 8;
    }

    void runppu(int x);

    inline int PaletteAdjustPixel(int pixel);

    LOGGER logger;
    void RENDER_LOG(int tmp) {
		if (logger.debug_loggingCD) {
			int addr = GetCHRAddress(tmp);
			if (addr != -1) {
				if (!(logger.cdloggervdata[addr] & 1)) {
					logger.cdloggervdata[addr] |= 1;
					if (logger.cdloggerVideoDataSize) {
						if (!(logger.cdloggervdata[addr] & 2)) undefinedvromcount--;
						rendercount++;
					}
				}
			}
		}
    }

    inline void Read(uint8& nt, uint8& pecnt, uint8& at, uint8 (&pt)[2]) {
	    NTRefreshAddr = RefreshAddr = ppur.get_ntread();
	    if (PEC586Hack)
		    ppur.s = (RefreshAddr & 0x200) >> 9;
	    pecnt = (RefreshAddr & 1) << 3;
        nt = Read(RefreshAddr);
	    runppu(kFetchTime);

	    RefreshAddr = ppur.get_atread();
	    at = Read(RefreshAddr);

	    //modify at to get appropriate palette shift
	    if (ppur.vt & 2) at >>= 4;
	    if (ppur.ht & 2) at >>= 2;
	    at &= 0x03;
	    at <<= 2;
	    //horizontal scroll clocked at cycle 3 and then
	    //vertical scroll at 251
	    runppu(1);
	    if (PPUON()) {
		    ppur.increment_hsc();
		    if (ppur.status.cycle == 251)
			    ppur.increment_vs();
	    }
	    runppu(1);

	    ppur.par = nt;
	    RefreshAddr = ppur.get_ptread();
	    if (PEC586Hack) {
		    if (ScreenON())
			    RENDER_LOG(RefreshAddr | pecnt);
		    pt[0] = Read(RefreshAddr | pecnt);
		    runppu(kFetchTime);
		    pt[1] = Read(RefreshAddr | pecnt);
		    runppu(kFetchTime);
	    } else {
		    if (ScreenON())
			    RENDER_LOG(RefreshAddr);
		    pt[0] = Read(RefreshAddr);
		    runppu(kFetchTime);
		    RefreshAddr |= 8;
		    if (ScreenON())
			    RENDER_LOG(RefreshAddr);
		    pt[1] = Read(RefreshAddr);
		    runppu(kFetchTime);
	    }
    }
};

}

#endif // define _PPUH
