#include "fds_obj.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "fceu_obj.h"
#include "sound_obj.h"

//	TODO:  Add code to put a delay in between the time a disk is inserted
//	and the when it can be successfully read/written to.  This should
//	prevent writes to wrong places OR add code to prevent disk ejects
//	when the virtual motor is on (mmm...virtual motor).

#define DC_INC    1

namespace fceu {

void FDS::FDSGI(GI h) {
	switch (h) {
	case GI_CLOSE: FDSClose(); break;
	case GI_POWER: FDSInit(); break;
	}
}

void FDS::FDSStateRestore(int version) {
	int x;

	fceu->cart.setmirror(((FDSRegs[5] & 8) >> 3) ^ 1);

	if (version >= 9810)
		for (x = 0; x < TotalSides; x++) {
			int b;
			for (b = 0; b < 65500; b++)
				diskdata[x][b] ^= diskdatao[x][b];
		}
}

void FDS::FDSInit(void) {
	memset(FDSRegs, 0, sizeof(FDSRegs));
	writeskip = DiskPtr = DiskSeekIRQ = 0;

	fceu->cart.setmirror(1);
	fceu->cart.setprg8(0xE000, 0);			// BIOS
	fceu->cart.setprg32r(1, 0x6000, 0);	// 32KB RAM
	fceu->cart.setchr8(0);					// 8KB CHR RAM

	fceu->x6502.MapIRQHook = &FDSFix_;
	fceu->GameStateRestore = &FDSStateRestore_;

	fceu->handler.SetReadHandler(0x4030, 0x4030, &FDSRead4030_);
	fceu->handler.SetReadHandler(0x4031, 0x4031, &FDSRead4031_);
	fceu->handler.SetReadHandler(0x4032, 0x4032, &FDSRead4032_);
	fceu->handler.SetReadHandler(0x4033, 0x4033, &FDSRead4033_);

	fceu->handler.SetWriteHandler(0x4020, 0x4025, &FDSWrite_);

	fceu->handler.SetWriteHandler(0x6000, 0xDFFF, &(fceu->cart.CartBW_));
	fceu->handler.SetReadHandler(0x6000, 0xFFFF, &(fceu->cart.CartBR_));

	IRQCount = IRQLatch = IRQa = 0;

	FDSSoundReset();
	InDisk = 0;
	SelectDisk = 0;
}

void FDS::FDSInsert(void)
{
	if (TotalSides == 0)
	{
        fceu::DispMessage("Not FDS; can't eject disk.", 0);
		return;
	}

	if (fceu->EmulationPaused())
		fceu->EmulationPaused_ |= EMULATIONPAUSED_FA;

	if (fceu->movie.Mode(MOVIEMODE_RECORD))
		fceu->movie.AddCommand(FCEUNPCMD_FDSINSERT);

	if (InDisk == 255)
	{
        fceu::DispMessage("Disk %d Side %s Inserted", 0, SelectDisk >> 1, (SelectDisk & 1) ? "B" : "A");
		InDisk = SelectDisk;
	} else
	{
        fceu::DispMessage("Disk %d Side %s Ejected", 0, SelectDisk >> 1, (SelectDisk & 1) ? "B" : "A");
		InDisk = 255;
	}
}
/*
void FCEU_FDSEject(void)
{
InDisk=255;
}
*/
void FDS::FDSSelect(void)
{
	if (TotalSides == 0)
	{
        fceu::DispMessage("Not FDS; can't select disk.", 0);
		return;
	}
	if (InDisk != 255)
	{
        fceu::DispMessage("Eject disk before selecting.", 0);
		return;
	}

	if (fceu->EmulationPaused())
		fceu->EmulationPaused_ |= EMULATIONPAUSED_FA;

	if (fceu->movie.Mode(MOVIEMODE_RECORD))
		fceu->movie.AddCommand(FCEUNPCMD_FDSSELECT);

	SelectDisk = ((SelectDisk + 1) % TotalSides) & 3;
    fceu::DispMessage("Disk %d Side %c Selected", 0, SelectDisk >> 1, (SelectDisk & 1) ? 'B' : 'A');
}

void FDS::FDSFix(int a) {
	if ((IRQa & 2) && IRQCount) {
		IRQCount -= a;
		if (IRQCount <= 0) {
			if (!(IRQa & 1)) {
				IRQa &= ~2;
				IRQCount = IRQLatch = 0;
			} else
				IRQCount = IRQLatch;
			fceu->x6502.IRQBegin(FCEU_IQEXT);
		}
	}
	if (DiskSeekIRQ > 0) {
		DiskSeekIRQ -= a;
		if (DiskSeekIRQ <= 0) {
			if (FDSRegs[5] & 0x80) {
				fceu->x6502.IRQBegin(FCEU_IQEXT2);
			}
		}
	}
}

uint8 FDS::FDSRead4030(uint32 A) {
	uint8 ret = 0;

	/* Cheap hack. */
	if (fceu->x6502.IRQlow & FCEU_IQEXT) ret |= 1;
	if (fceu->x6502.IRQlow & FCEU_IQEXT2) ret |= 2;

	if (!fceu->ppu.get_fceuindbg()) {
		fceu->x6502.IRQEnd(FCEU_IQEXT);
		fceu->x6502.IRQEnd(FCEU_IQEXT2);
	}
	return ret;
}

uint8 FDS::FDSRead4031(uint32 A) {
	static uint8 z = 0;
	if (InDisk != 255) {
		z = diskdata[InDisk][DiskPtr];
		if (!fceu->ppu.get_fceuindbg()) {
			if (DiskPtr < 64999) DiskPtr++;
			DiskSeekIRQ = 150;
			fceu->x6502.IRQEnd(FCEU_IQEXT2);
		}
	}
	return z;
}
uint8 FDS::FDSRead4032(uint32 A) {
	uint8 ret;

	ret = fceu->x6502.DB_ & ~7;
	if (InDisk == 255)
		ret |= 5;

	if (InDisk == 255 || !(FDSRegs[5] & 1) || (FDSRegs[5] & 2))
		ret |= 2;
	return ret;
}

uint8 FDS::FDSRead4033(uint32 A) {
	return 0x80; // battery
}

/* Begin FDS sound */

#define FDSClock (1789772.7272727272727272 / 2)

#define  SPSG  fdso.SPSG
#define b19shiftreg60  fdso.b19shiftreg60
#define b24adder66  fdso.b24adder66
#define b24latch68  fdso.b24latch68
#define b17latch76  fdso.b17latch76
#define b8shiftreg88  fdso.b8shiftreg88
#define clockcount  fdso.clockcount
#define amplitude  fdso.amplitude
#define speedo    fdso.speedo

void FDS::FDSSoundStateAdd(void) {
	AddExState(fdso.cwave, 64, 0, "WAVE");
	AddExState(fdso.mwave, 32, 0, "MWAV");
	AddExState(amplitude, 2, 0, "AMPL");
	AddExState(SPSG, 0xB, 0, "SPSG");

	AddExState(&b8shiftreg88, 1, 0, "B88");

	AddExState(&clockcount, 4, 1, "CLOC");
	AddExState(&b19shiftreg60, 4, 1, "B60");
	AddExState(&b24adder66, 4, 1, "B66");
	AddExState(&b24latch68, 4, 1, "B68");
	AddExState(&b17latch76, 4, 1, "B76");
}

uint8 FDS::FDSSRead(uint32 A) {
	switch (A & 0xF) {
	case 0x0: return(amplitude[0] | (fceu->x6502.DB_ & 0xC0));
	case 0x2: return(amplitude[1] | (fceu->x6502.DB_ & 0xC0));
	}
	return(fceu->x6502.DB_);
}

void FDS::FDSSWrite(uint32 A, uint8 V) {
	if (fceu->FSettings.SndRate) {
		if (fceu->FSettings.soundq >= 1)
			RenderSoundHQ();
		else
			RenderSound();
	}
	A -= 0x4080;
	switch (A) {
	case 0x0:
	case 0x4:
		if (V & 0x80)
			amplitude[(A & 0xF) >> 2] = V & 0x3F;
		break;
	case 0x7:
		b17latch76 = 0;
		SPSG[0x5] = 0;
		break;
	case 0x8:
		b17latch76 = 0;
		fdso.mwave[SPSG[0x5] & 0x1F] = V & 0x7;
		SPSG[0x5] = (SPSG[0x5] + 1) & 0x1F;
		break;
	}
	SPSG[A] = V;
}

// $4080 - Fundamental wave amplitude data register 92
// $4082 - Fundamental wave frequency data register 58
// $4083 - Same as $4082($4083 is the upper 4 bits).

// $4084 - Modulation amplitude data register 78
// $4086 - Modulation frequency data register 72
// $4087 - Same as $4086($4087 is the upper 4 bits)


void FDS::DoEnv() {
	int x;

	for (x = 0; x < 2; x++)
		if (!(SPSG[x << 2] & 0x80) && !(SPSG[0x3] & 0x40)) {
			static int counto[2] = { 0, 0 };

			if (counto[x] <= 0) {
				if (!(SPSG[x << 2] & 0x80)) {
					if (SPSG[x << 2] & 0x40) {
						if (amplitude[x] < 0x3F)
							amplitude[x]++;
					} else {
						if (amplitude[x] > 0)
							amplitude[x]--;
					}
				}
				counto[x] = (SPSG[x << 2] & 0x3F);
			} else
				counto[x]--;
		}
}

uint8 FDS::FDSWaveRead(uint32 A) {
	return(fdso.cwave[A & 0x3f] | (fceu->x6502.DB_ & 0xC0));
}

void FDS::FDSWaveWrite(uint32 A, uint8 V) {
	if (SPSG[0x9] & 0x80)
		fdso.cwave[A & 0x3f] = V & 0x3F;
}

inline void FDS::ClockRise(void) {
	if (!clockcount) {
		ta++;

		b19shiftreg60 = (SPSG[0x2] | ((SPSG[0x3] & 0xF) << 8));
		b17latch76 = (SPSG[0x6] | ((SPSG[0x07] & 0xF) << 8)) + b17latch76;

		if (!(SPSG[0x7] & 0x80)) {
			int t = fdso.mwave[(b17latch76 >> 13) & 0x1F] & 7;
			int t2 = amplitude[1];
			int adj = 0;

			if ((t & 3)) {
				if ((t & 4))
					adj -= (t2 * ((4 - (t & 3))));
				else
					adj += (t2 * ((t & 3)));
			}
			adj *= 2;
			if (adj > 0x7F) adj = 0x7F;
			if (adj < -0x80) adj = -0x80;
			b8shiftreg88 = 0x80 + adj;
		} else {
			b8shiftreg88 = 0x80;
		}
	} else {
		b19shiftreg60 <<= 1;
		b8shiftreg88 >>= 1;
	}
	b24adder66 = (b24latch68 + b19shiftreg60) & 0x1FFFFFF;
}

inline void FDS::ClockFall(void) {
	if ((b8shiftreg88 & 1))
		b24latch68 = b24adder66;
	clockcount = (clockcount + 1) & 7;
}

inline int32 FDS::FDSDoSound(void) {
	fdso.count += fdso.cycles;
	if (fdso.count >= ((int64)1 << 40)) {
 dogk:
		fdso.count -= (int64)1 << 40;
		ClockRise();
		ClockFall();
		fdso.envcount--;
		if (fdso.envcount <= 0) {
			fdso.envcount += SPSG[0xA] * 3;
			DoEnv();
		}
	}
	if (fdso.count >= 32768) goto dogk;

	// Might need to emulate applying the amplitude to the waveform a bit better...
	{
		int k = amplitude[0];
		if (k > 0x20) k = 0x20;
		return (fdso.cwave[b24latch68 >> 19] * k) * 4 / ((SPSG[0x9] & 0x3) + 2);
	}
}

void FDS::RenderSound(void) {
	int32 end, start;
	int32 x;

	start = FBC;
	end = (SoundTimestamp() << 16) / soundtsinc;
	if (end <= start)
		return;
	FBC = end;

	if (!(SPSG[0x9] & 0x80))
		for (x = start; x < end; x++) {
			uint32 t = FDSDoSound();
			t += t >> 1;
			t >>= 4;
			Wave[x >> 4] += t; //(t>>2)-(t>>3); //>>3;
		}
}

void FDS::RenderSoundHQ(void) {
	uint32 x; //mbg merge 7/17/06 - made this unsigned

	if (!(SPSG[0x9] & 0x80))
		for (x = FBC; x < SoundTimestamp(); x++) {
			uint32 t = FDSDoSound();
			t += t >> 1;
			WaveHi[x] += t; //(t<<2)-(t<<1);
		}
	FBC = SoundTimestamp();
}

void FDS::HQSync(int32 ts) {
	FBC = ts;
}

void FDS::FDSSound(int c) {
	RenderSound();
	FBC = c;
}

void FDS::FDS_ESI(void) {
	if (fceu->FSettings.SndRate) {
		if (fceu->FSettings.soundq >= 1) {
			fdso.cycles = (int64)1 << 39;
		} else {
			fdso.cycles = ((int64)1 << 40) * FDSClock;
			fdso.cycles /= fceu->FSettings.SndRate * 16;
		}
	}
	fceu->handler.SetReadHandler(0x4040, 0x407f, &FDSWaveRead_);
	fceu->handler.SetWriteHandler(0x4040, 0x407f, &FDSWaveWrite_);
	fceu->handler.SetWriteHandler(0x4080, 0x408A, &FDSSWrite_);
	fceu->handler.SetReadHandler(0x4090, 0x4092, &FDSSRead_);
}

void FDS::FDSSoundReset(void) {
	memset(&fdso, 0, sizeof(fdso));
	FDS_ESI();
	GameExpSound.HiSync = &HQSync_;
	GameExpSound.HiFill = &RenderSoundHQ_;
	GameExpSound.Fill = &FDSSound_;
	GameExpSound.RChange = &FDS_ESI_;
}

void FDS::FDSWrite(uint32 A, uint8 V) {
	switch (A) {
	case 0x4020:
		fceu->x6502.IRQEnd(FCEU_IQEXT);
		IRQLatch &= 0xFF00;
		IRQLatch |= V;
		break;
	case 0x4021:
		fceu->x6502.IRQEnd(FCEU_IQEXT);
		IRQLatch &= 0xFF;
		IRQLatch |= V << 8;
		break;
	case 0x4022:
		fceu->x6502.IRQEnd(FCEU_IQEXT);
		IRQCount = IRQLatch;
		IRQa = V & 3;
		break;
	case 0x4023: break;
	case 0x4024:
		if ((InDisk != 255) && !(FDSRegs[5] & 0x4) && (FDSRegs[3] & 0x1)) {
			if (DiskPtr >= 0 && DiskPtr < 65500) {
				if (writeskip)
					writeskip--;
				else if (DiskPtr >= 2) {
					DiskWritten = 1;
					diskdata[InDisk][DiskPtr - 2] = V;
				}
			}
		}
		break;
	case 0x4025:
		fceu->x6502.IRQEnd(FCEU_IQEXT2);
		if (InDisk != 255) {
			if (!(V & 0x40)) {
				if ((FDSRegs[5] & 0x40) && !(V & 0x10)) {
					DiskSeekIRQ = 200;
					DiskPtr -= 2;
				}
				if (DiskPtr < 0) DiskPtr = 0;
			}
			if (!(V & 0x4)) writeskip = 2;
			if (V & 2) {
				DiskPtr = 0; DiskSeekIRQ = 200;
			}
			if (V & 0x40) DiskSeekIRQ = 200;
		}
		fceu->cart.setmirror(((V >> 3) & 1) ^ 1);
		break;
	}
	FDSRegs[A & 7] = V;
}

void FDS::FreeFDSMemory(void) {
	int x;

	for (x = 0; x < TotalSides; x++)
		if (diskdata[x]) {
			free(diskdata[x]);
			diskdata[x] = 0;
		}
}

int FDS::SubLoad(FCEUFILE *fp) {
	struct md5_context md5;
	uint8 header[16];
	int x;

    fceu::fread(header, 16, 1, fp);

	if (memcmp(header, "FDS\x1a", 4)) {
		if (!(memcmp(header + 1, "*NINTENDO-HVC*", 14))) {
			long t;
			t = fceu::fgetsize(fp);
			if (t < 65500)
				t = 65500;
			TotalSides = t / 65500;
            fceu::fseek(fp, 0, SEEK_SET);
		} else
			return(0);
	} else
		TotalSides = header[4];

	md5_starts(&md5);

	if (TotalSides > 8) TotalSides = 8;
	if (TotalSides < 1) TotalSides = 1;

	for (x = 0; x < TotalSides; x++) {
		diskdata[x] = (uint8*)fceu::malloc(65500);
		if (!diskdata[x]) {
			int zol;
			for (zol = 0; zol < x; zol++)
				free(diskdata[zol]);
			return 0;
		}
        fceu::fread(diskdata[x], 1, 65500, fp);
		md5_update(&md5, diskdata[x], 65500);
	}
	md5_finish(&md5, fceu->GameInfo->MD5.data);
	return(1);
}

void FDS::PreSave(void) {
	int x;
	for (x = 0; x < TotalSides; x++) {
		int b;
		for (b = 0; b < 65500; b++)
			diskdata[x][b] ^= diskdatao[x][b];
	}
}

void FDS::PostSave(void) {
	int x;
	for (x = 0; x < TotalSides; x++) {
		int b;
		for (b = 0; b < 65500; b++)
			diskdata[x][b] ^= diskdatao[x][b];
	}
}

int FDS::FDSLoad(const char *name, FCEUFILE *fp) {
	FILE *zp;
	int x;

	char *fn = strdup(fceu::MakeFName(FCEUMKF_FDSROM, 0, 0).c_str());

	if (!(zp = fceu::UTF8fopen(fn, "rb"))) {
            fceu::PrintError("FDS BIOS ROM image missing: %s", fceu::MakeFName(FCEUMKF_FDSROM, 0, 0).c_str());
		free(fn);
		return 0;
	}

	free(fn);

	fseek(zp, 0L, SEEK_END);
	if (ftell(zp) != 8192) {
		fclose(zp);
        fceu::PrintError("FDS BIOS ROM image incompatible: %s", fceu::MakeFName(FCEUMKF_FDSROM, 0, 0).c_str());
		return 0;
	}
	fseek(zp, 0L, SEEK_SET);

	fceu->cart.ResetCartMapping();

	if(FDSBIOS)
		free(FDSBIOS);
	FDSBIOS = NULL;
	if(FDSRAM)
		free(FDSRAM);
	FDSRAM = NULL;
	if(CHRRAM)
		free(CHRRAM);
	CHRRAM = NULL;

	FDSBIOSsize = 8192;
	FDSBIOS = (uint8*)fceu::gmalloc(FDSBIOSsize);
	fceu->cart.SetupCartPRGMapping(0, FDSBIOS, FDSBIOSsize, 0);

	if (fread(FDSBIOS, 1, FDSBIOSsize, zp) != FDSBIOSsize) {
		if(FDSBIOS)
			free(FDSBIOS);
		FDSBIOS = NULL;
		fclose(zp);
        fceu::PrintError("Error reading FDS BIOS ROM image.");
		return 0;
	}

	fclose(zp);

    fceu::fseek(fp, 0, SEEK_SET);

	FreeFDSMemory();
	if (!SubLoad(fp)) {
		if(FDSBIOS)
			free(FDSBIOS);
		FDSBIOS = NULL;
		return(0);
	}

	if (!fceu->cart.GetDisableBatteryLoading()) {
		FCEUFILE *tp;
		char *fn = strdup(fceu::MakeFName(FCEUMKF_FDS, 0, 0).c_str());

		int x;
		for (x = 0; x < TotalSides; x++) {
			diskdatao[x] = (uint8*)fceu::malloc(65500);
			memcpy(diskdatao[x], diskdata[x], 65500);
		}

		if ((tp = fceu::fopen(fn, 0, "rb", 0))) {
            fceu::printf("Disk was written. Auxillary FDS file open \"%s\".\n",fn);
			FreeFDSMemory();
			if (!SubLoad(tp)) {
                fceu::PrintError("Error reading auxillary FDS file.");
				if(FDSBIOS)
					free(FDSBIOS);
				FDSBIOS = NULL;
				free(fn);
				return(0);
			}
            fceu::fclose(tp);
			DiskWritten = 1;  /* For save state handling. */
		}
		free(fn);
	}

	extern char LoadedRomFName[2048];
	strcpy(LoadedRomFName, name); //For the debugger list

	fceu->GameInfo->type = GIT_FDS;
	fceu->GameInterface = &FDSGI_;
	isFDS = true;

	SelectDisk = 0;
	InDisk = 255;

	ResetExState(&PreSave_, &PostSave_);
	FDSSoundStateAdd();

	for (x = 0; x < TotalSides; x++) {
		char temp[5];
		sprintf(temp, "DDT%d", x);
		AddExState(diskdata[x], 65500, 0, temp);
	}

	AddExState(FDSRegs, sizeof(FDSRegs), 0, "FREG");
	AddExState(&IRQCount, 4, 1, "IRQC");
	AddExState(&IRQLatch, 4, 1, "IQL1");
	AddExState(&IRQa, 1, 0, "IRQA");
	AddExState(&writeskip, 1, 0, "WSKI");
	AddExState(&DiskPtr, 4, 1, "DPTR");
	AddExState(&DiskSeekIRQ, 4, 1, "DSIR");
	AddExState(&SelectDisk, 1, 0, "SELD");
	AddExState(&InDisk, 1, 0, "INDI");
	AddExState(&DiskWritten, 1, 0, "DSKW");

	CHRRAMSize = 8192;
	CHRRAM = (uint8*)fceu::gmalloc(CHRRAMSize);
	memset(CHRRAM, 0, CHRRAMSize);
	fceu->cart.SetupCartCHRMapping(0, CHRRAM, CHRRAMSize, 1);
	AddExState(CHRRAM, CHRRAMSize, 0, "CHRR");

	FDSRAMSize = 32768;
	FDSRAM = (uint8*)fceu::gmalloc(FDSRAMSize);
	memset(FDSRAM, 0, FDSRAMSize);
	fceu->cart.SetupCartPRGMapping(1, FDSRAM, FDSRAMSize, 1);
	AddExState(FDSRAM, FDSRAMSize, 0, "FDSR");

	fceu->cart.SetupCartMirroring(0, 0, 0);

    fceu::printf(" Sides: %d\n\n", TotalSides);

	fceu->SetVidSystem(0);

	return 1;
}

void FDS::FDSClose(void) {
	FILE *fp;
	int x;
	isFDS = false;

	if (!DiskWritten) return;

	const std::string &fn = fceu::MakeFName(FCEUMKF_FDS, 0, 0);
	if (!(fp = fceu::UTF8fopen(fn.c_str(), "wb"))) {
		return;
	}

	for (x = 0; x < TotalSides; x++) {
		if (fwrite(diskdata[x], 1, 65500, fp) != 65500) {
            fceu::PrintError("Error saving FDS image!");
			fclose(fp);
			return;
		}
	}

	for (x = 0; x < TotalSides; x++)
		if (diskdatao[x]) {
			free(diskdatao[x]);
			diskdatao[x] = 0;
		}

	FreeFDSMemory();
	if(FDSBIOS)
		free(FDSBIOS);
	FDSBIOS = NULL;
	if(FDSRAM)
		free(FDSRAM);
	FDSRAM = NULL;
	if(CHRRAM)
		free(CHRRAM);
	CHRRAM = NULL;
	fclose(fp);
}

} // namespace fceu
