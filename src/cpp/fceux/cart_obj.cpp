#include "cart_obj.h"

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <climits>

#include "fceu_obj.h"

#include "utils/general_obj.h"

namespace fceu {

inline void Cart::setpageptr(int s, uint32 A, uint8 *p, int ram) {
	uint32 AB = A >> 11;
	int x;

	if (p)
		for (x = (s >> 1) - 1; x >= 0; x--) {
			PRGIsRAM[AB + x] = ram;
			Page[AB + x] = p - A;
		}
	else
		for (x = (s >> 1) - 1; x >= 0; x--) {
			PRGIsRAM[AB + x] = 0;
			Page[AB + x] = 0;
		}
}

void Cart::ResetCartMapping(void) {
	int x;

	fceu->ppu.ResetHooks();

	for (x = 0; x < 32; x++) {
		Page[x] = nothing - x * 2048;
		PRGptr[x] = CHRptr[x] = 0;
		PRGsize[x] = CHRsize[x] = 0;
	}
	for (x = 0; x < 8; x++) {
		MMC5SPRVPage[x] = MMC5BGVPage[x] = VPageR[x] = nothing - 0x400 * x;
	}
}

void Cart::SetupCartPRGMapping(int chip, uint8 *p, uint32 size, int ram) {
	PRGptr[chip] = p;
	PRGsize[chip] = size;

	PRGmask2[chip] = (size >> 11) - 1;
	PRGmask4[chip] = (size >> 12) - 1;
	PRGmask8[chip] = (size >> 13) - 1;
	PRGmask16[chip] = (size >> 14) - 1;
	PRGmask32[chip] = (size >> 15) - 1;

	PRGram[chip] = ram ? 1 : 0;
}

void Cart::SetupCartCHRMapping(int chip, uint8 *p, uint32 size, int ram) {
	CHRptr[chip] = p;
	CHRsize[chip] = size;

	CHRmask1[chip] = (size >> 10) - 1;
	CHRmask2[chip] = (size >> 11) - 1;
	CHRmask4[chip] = (size >> 12) - 1;
	CHRmask8[chip] = (size >> 13) - 1;

	CHRram[chip] = ram;
}

uint8 Cart::CartBR(uint32 A) {
	return Page[A >> 11][A];
}

void Cart::CartBW(uint32 A, uint8 V) {
	//printf("Ok: %04x:%02x, %d\n",A,V,PRGIsRAM[A>>11]);
	if (PRGIsRAM[A >> 11] && Page[A >> 11])
		Page[A >> 11][A] = V;
}

uint8 Cart::CartBROB(uint32 A) {
	if (!Page[A >> 11])
		return(fceu->x6502.DB());
	else
		return Page[A >> 11][A];
}

void Cart::setprg2r(int r, uint32 A, uint32 V) {
	V &= PRGmask2[r];
	setpageptr(2, A, PRGptr[r] ? (&PRGptr[r][V << 11]) : 0, PRGram[r]);
}

void Cart::setprg2(uint32 A, uint32 V) {
	setprg2r(0, A, V);
}

void Cart::setprg4r(int r, uint32 A, uint32 V) {
	V &= PRGmask4[r];
	setpageptr(4, A, PRGptr[r] ? (&PRGptr[r][V << 12]) : 0, PRGram[r]);
}

void Cart::setprg4(uint32 A, uint32 V) {
	setprg4r(0, A, V);
}

void Cart::setprg8r(int r, uint32 A, uint32 V) {
	if (PRGsize[r] >= 8192) {
		V &= PRGmask8[r];
		setpageptr(8, A, PRGptr[r] ? (&PRGptr[r][V << 13]) : 0, PRGram[r]);
	} else {
		uint32 VA = V << 2;
		int x;
		for (x = 0; x < 4; x++)
			setpageptr(2, A + (x << 11), PRGptr[r] ? (&PRGptr[r][((VA + x) & PRGmask2[r]) << 11]) : 0, PRGram[r]);
	}
}

void Cart::setprg8(uint32 A, uint32 V) {
	setprg8r(0, A, V);
}

void Cart::setprg16r(int r, uint32 A, uint32 V) {
	if (PRGsize[r] >= 16384) {
		V &= PRGmask16[r];
		setpageptr(16, A, PRGptr[r] ? (&PRGptr[r][V << 14]) : 0, PRGram[r]);
	} else {
		uint32 VA = V << 3;
		int x;

		for (x = 0; x < 8; x++)
			setpageptr(2, A + (x << 11), PRGptr[r] ? (&PRGptr[r][((VA + x) & PRGmask2[r]) << 11]) : 0, PRGram[r]);
	}
}

void Cart::setprg16(uint32 A, uint32 V) {
	setprg16r(0, A, V);
}

void Cart::setprg32r(int r, uint32 A, uint32 V) {
	if (PRGsize[r] >= 32768) {
		V &= PRGmask32[r];
		setpageptr(32, A, PRGptr[r] ? (&PRGptr[r][V << 15]) : 0, PRGram[r]);
	} else {
		uint32 VA = V << 4;
		int x;

		for (x = 0; x < 16; x++)
			setpageptr(2, A + (x << 11), PRGptr[r] ? (&PRGptr[r][((VA + x) & PRGmask2[r]) << 11]) : 0, PRGram[r]);
	}
}

void Cart::setprg32(uint32 A, uint32 V) {
	setprg32r(0, A, V);
}

void Cart::setchr1r(int r, uint32 A, uint32 V) {
	if (!CHRptr[r]) return;
	fceu->ppu.LineUpdate();
	V &= CHRmask1[r];
	if (CHRram[r])
		fceu->ppu.PPUCHRRAM |= (1 << (A >> 10));
	else
		fceu->ppu.PPUCHRRAM &= ~(1 << (A >> 10));
	VPageR[(A) >> 10] = &CHRptr[r][(V) << 10] - (A);
}

void Cart::setchr2r(int r, uint32 A, uint32 V) {
	if (!CHRptr[r]) return;
	fceu->ppu.LineUpdate();
	V &= CHRmask2[r];
	VPageR[(A) >> 10] = VPageR[((A) >> 10) + 1] = &CHRptr[r][(V) << 11] - (A);
	if (CHRram[r])
		fceu->ppu.PPUCHRRAM |= (3 << (A >> 10));
	else
		fceu->ppu.PPUCHRRAM &= ~(3 << (A >> 10));
}

void Cart::setchr4r(int r, unsigned int A, unsigned int V) {
	if (!CHRptr[r]) return;
	fceu->ppu.LineUpdate();
	V &= CHRmask4[r];
	VPageR[(A) >> 10] = VPageR[((A) >> 10) + 1] =
							VPageR[((A) >> 10) + 2] = VPageR[((A) >> 10) + 3] = &CHRptr[r][(V) << 12] - (A);
	if (CHRram[r])
		fceu->ppu.PPUCHRRAM |= (15 << (A >> 10));
	else
		fceu->ppu.PPUCHRRAM &= ~(15 << (A >> 10));
}

void Cart::setchr8r(int r, uint32 V) {
	int x;

	if (!CHRptr[r]) return;
	fceu->ppu.LineUpdate();
	V &= CHRmask8[r];
	for (x = 7; x >= 0; x--)
		VPageR[x] = &CHRptr[r][V << 13];
	if (CHRram[r])
		fceu->ppu.PPUCHRRAM |= (255);
	else
		fceu->ppu.PPUCHRRAM = 0;
}

void Cart::setchr1(uint32 A, uint32 V) {
	setchr1r(0, A, V);
}

void Cart::setchr2(uint32 A, uint32 V) {
	setchr2r(0, A, V);
}

void Cart::setchr4(uint32 A, uint32 V) {
	setchr4r(0, A, V);
}

void Cart::setchr8(uint32 V) {
	setchr8r(0, V);
}

void Cart::setntamem(uint8 *p, int ram, uint32 b) {
	fceu->ppu.LineUpdate();
	fceu->ppu.vnapage[b] = p;
	fceu->ppu.PPUNTARAM &= ~(1 << b);
	if (ram)
		fceu->ppu.PPUNTARAM |= 1 << b;
}

void Cart::setmirrorw(int a, int b, int c, int d) {
	fceu->ppu.LineUpdate();
    uint8** vnapage = fceu->ppu.vnapage;
	vnapage[0] = fceu->ppu.NTARAM + a * 0x400;
	vnapage[1] = fceu->ppu.NTARAM + b * 0x400;
	vnapage[2] = fceu->ppu.NTARAM + c * 0x400;
	vnapage[3] = fceu->ppu.NTARAM + d * 0x400;
}

void Cart::setmirror(int t) {
	fceu->ppu.LineUpdate();
	if (!mirrorhard) {
        uint8** vnapage = fceu->ppu.vnapage;
		switch (t) {
		case MI_H:
			vnapage[0] = vnapage[1] = fceu->ppu.NTARAM; vnapage[2] = vnapage[3] = fceu->ppu.NTARAM + 0x400;
			break;
		case MI_V:
			vnapage[0] = vnapage[2] = fceu->ppu.NTARAM; vnapage[1] = vnapage[3] = fceu->ppu.NTARAM + 0x400;
			break;
		case MI_0:
			vnapage[0] = vnapage[1] = vnapage[2] = vnapage[3] = fceu->ppu.NTARAM;
			break;
		case MI_1:
			vnapage[0] = vnapage[1] = vnapage[2] = vnapage[3] = fceu->ppu.NTARAM + 0x400;
			break;
		}
		fceu->ppu.PPUNTARAM = 0xF;
	}
}

void Cart::SetupCartMirroring(int m, int hard, uint8 *extra) {
	if (m < 4) {
		mirrorhard = 0;
		setmirror(m);
	} else {
		fceu->ppu.vnapage[0] = fceu->ppu.NTARAM;
		fceu->ppu.vnapage[1] = fceu->ppu.NTARAM + 0x400;
		fceu->ppu.vnapage[2] = extra;
		fceu->ppu.vnapage[3] = extra + 0x400;
		fceu->ppu.PPUNTARAM = 0xF;
	}
	mirrorhard = hard;
}

// Called when a game(file) is opened successfully. Returns TRUE on error.
bool Cart::OpenGenie(void)
{
	FILE *fp;
	int x;

	if (!GENIEROM)
	{
		char *fn;

		if (!(GENIEROM = (uint8*)fceu::malloc(4096 + 1024)))
			return true;

		fn = strdup(MakeFName(FCEUMKF_GGROM, 0, 0).c_str());
		fp = fceu::UTF8fopen(fn, "rb");
		if (!fp)
		{
            fceu::PrintError("Error opening Game Genie ROM image!\nIt should be named \"gg.rom\"!");
			free(GENIEROM);
			GENIEROM = 0;
			return true;
		}
		if (fread(GENIEROM, 1, 16, fp) != 16)
		{
 grerr:
            fceu::PrintError("Error reading from Game Genie ROM image!");
			free(GENIEROM);
			GENIEROM = 0;
			fclose(fp);
			return true;
		}
		if (GENIEROM[0] == 0x4E)
		{
			/* iNES ROM image */
			if (fread(GENIEROM, 1, 4096, fp) != 4096)
				goto grerr;
			if (fseek(fp, 16384 - 4096, SEEK_CUR))
				goto grerr;
			if (fread(GENIEROM + 4096, 1, 256, fp) != 256)
				goto grerr;
		} else
		{
			if (fread(GENIEROM + 16, 1, 4352 - 16, fp) != (4352 - 16))
				goto grerr;
		}
		fclose(fp);

		/* Workaround for the FCE Ultra CHR page size only being 1KB */
		for (x = 0; x < 4; x++)
		{
			memcpy(GENIEROM + 4096 + (x << 8), GENIEROM + 4096, 256);
		}
	}

	geniestage = 1;
	return false;
}

/* Called when a game is closed. */
void Cart::CloseGenie(void) {
	/* No good reason to free() the Game Genie ROM image data. */
	geniestage = 0;
	fceu->handler.FlushGenieRW();
	VPageR = VPage;
}

void Cart::KillGenie(void) {
	if (GENIEROM) {
		free(GENIEROM);
		GENIEROM = 0;
	}
}

uint8 Cart::GenieRead(uint32 A) {
	return GENIEROM[A & 4095];
}

void Cart::GenieWrite(uint32 A, uint8 V) {
	switch (A) {
	case 0x800c:
	case 0x8008:
	case 0x8004: genieval[((A - 4) & 0xF) >> 2] = V; break;

	case 0x800b:
	case 0x8007:
	case 0x8003: geniech[((A - 3) & 0xF) >> 2] = V; break;

	case 0x800a:
	case 0x8006:
	case 0x8002: genieaddr[((A - 2) & 0xF) >> 2] &= 0xFF00; genieaddr[((A - 2) & 0xF) >> 2] |= V; break;

	case 0x8009:
	case 0x8005:
	case 0x8001: genieaddr[((A - 1) & 0xF) >> 2] &= 0xFF; genieaddr[((A - 1) & 0xF) >> 2] |= (V | 0x80) << 8; break;

	case 0x8000:
		if (!V)
			FixGenieMap();
		else {
			modcon = V ^ 0xFF;
			if (V == 0x71)
				modcon = 0;
		}
		break;
	}
}

uint8 Cart::GenieFix1(uint32 A) {
	uint8 r = (*GenieBackup[0])(A);

	if ((modcon >> 1) & 1) // No check
		return genieval[0];
	else if (r == geniech[0])
		return genieval[0];

	return r;
}

uint8 Cart::GenieFix2(uint32 A) {
	uint8 r = (*GenieBackup[1])(A);

	if ((modcon >> 2) & 1) // No check
		return genieval[1];
	else if (r == geniech[1])
		return genieval[1];

	return r;
}

uint8 Cart::GenieFix3(uint32 A) {
	uint8 r = (*GenieBackup[2])(A);

	if ((modcon >> 3) & 1) // No check
		return genieval[2];
	else if (r == geniech[2])
		return genieval[2];

	return r;
}

void Cart::FixGenieMap(void) {
	int x;

	geniestage = 2;

	for (x = 0; x < 8; x++)
		VPage[x] = VPageG[x];

	VPageR = VPage;
	fceu->handler.FlushGenieRW();
	//printf("Rightyo\n");
	for (x = 0; x < 3; x++)
		if ((modcon >> (4 + x)) & 1) {
			readfunc* tmp[3] = { &GenieFix1_, &GenieFix2_, &GenieFix3_ };
			GenieBackup[x] = fceu->handler.GetReadHandler(genieaddr[x]);
			fceu->handler.SetReadHandler(genieaddr[x], genieaddr[x], tmp[x]);
		}
}

void Cart::GeniePower(void) {
	uint32 x;

	if (!geniestage)
		return;

	geniestage = 1;
	for (x = 0; x < 3; x++) {
		genieval[x] = 0xFF;
		geniech[x] = 0xFF;
		genieaddr[x] = 0xFFFF;
	}
	modcon = 0;

	fceu->handler.SetWriteHandler(0x8000, 0xFFFF, &GenieWrite_);
	fceu->handler.SetReadHandler(0x8000, 0xFFFF, &GenieRead_);

	for (x = 0; x < 8; x++)
		VPage[x] = GENIEROM + 4096 - 0x400 * x;

	if (fceu->handler.AllocGenieRW())
		VPageR = VPageG;
	else
		geniestage = 2;
}

void Cart::SaveGameSave(CartInfo *LocalHWInfo) {
	if (LocalHWInfo->battery && LocalHWInfo->SaveGame[0]) {
		FILE *sp;

		std::string soot = MakeFName(FCEUMKF_SAV, 0, "sav");
		if ((sp = fceu::UTF8fopen(soot, "wb")) == NULL) {
			fceu::PrintError("WRAM file \"%s\" cannot be written to.\n", soot.c_str());
		} else {
			for (int x = 0; x < 4; x++)
				if (LocalHWInfo->SaveGame[x]) {
					fwrite(LocalHWInfo->SaveGame[x], 1,
						   LocalHWInfo->SaveGameLen[x], sp);
				}
		}
	}
}

void Cart::LoadGameSave(CartInfo *LocalHWInfo) {
	if (LocalHWInfo->battery && LocalHWInfo->SaveGame[0] && !disableBatteryLoading) {
		FILE *sp;

		std::string soot = MakeFName(FCEUMKF_SAV, 0, "sav");
		sp = fceu::UTF8fopen(soot, "rb");
		if (sp != NULL) {
			for (int x = 0; x < 4; x++)
				if (LocalHWInfo->SaveGame[x])
					fread(LocalHWInfo->SaveGame[x], 1, LocalHWInfo->SaveGameLen[x], sp);
		}
	}
}

//clears all save memory. call this if you want to pretend the saveram has been reset (it doesnt touch what is on disk though)
void Cart::ClearGameSave(CartInfo *LocalHWInfo) {
	if (LocalHWInfo->battery && LocalHWInfo->SaveGame[0]) {
		for (int x = 0; x < 4; x++)
			if (LocalHWInfo->SaveGame[x])
				memset(LocalHWInfo->SaveGame[x], 0, LocalHWInfo->SaveGameLen[x]);
	}
}

} // namespace fceu
