/* FCE Ultra - NES/Famicom Emulator
 *
 * Copyright notice for this file:
 *  Copyright (C) 1998 BERO
 *  Copyright (C) 2003 Xodnizel
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 */

#include "ppu_obj.h"

namespace FCEU {

void PPU::makeppulut(void) {
	int x;
	int y;
	int cc, xo, pixel;


	for (x = 0; x < 256; x++) {
		ppulut1[x] = 0;
		for (y = 0; y < 8; y++)
			ppulut1[x] |= ((x >> (7 - y)) & 1) << (y * 4);
		ppulut2[x] = ppulut1[x] << 1;
	}

	for (cc = 0; cc < 16; cc++) {
		for (xo = 0; xo < 8; xo++) {
			ppulut3[xo | (cc << 3)] = 0;
			for (pixel = 0; pixel < 8; pixel++) {
				int shiftr;
				shiftr = (pixel + xo) / 8;
				shiftr *= 2;
				ppulut3[xo | (cc << 3)] |= ((cc >> shiftr) & 3) << (2 + pixel * 4);
			}
		}
	}
}

//this duplicates logic which is embedded in the ppu rendering code
//which figures out where to get CHR data from depending on various hack modes
//mostly involving mmc5.
//this might be incomplete.
uint8* PPU::GetCHR(uint32 vadr, uint32 refreshaddr) {
	if (MMC5Hack) {
		if (MMC5HackCHRMode == 1) {
			uint8 *C = MMC5HackVROMPTR;
			C += (((MMC5HackExNTARAMPtr[refreshaddr & 0x3ff]) & 0x3f & MMC5HackVROMMask) << 12) + (vadr & 0xfff);
			C += (MMC50x5130 & 0x3) << 18;	//11-jun-2009 for kuja_killer
			return C;
		} else {
			return MMC5BGVRAMADR(vadr);
		}
	} else return VRAMADR(vadr);
}

//likewise for ATTR
int PPU::GetAttr(int ntnum, int xt, int yt) {
	int attraddr = 0x3C0 + ((yt >> 2) << 3) + (xt >> 2);
	int temp = (((yt & 2) << 1) + (xt & 2));
	int refreshaddr = xt + yt * 32;
	if (MMC5Hack && MMC5HackCHRMode == 1)
		return (MMC5HackExNTARAMPtr[refreshaddr & 0x3ff] & 0xC0) >> 6;
	else
		return (vnapage[ntnum][attraddr] & (3 << temp)) >> temp;
}

//new ppu-----
void PPU::Write_Default(uint32 A, uint8 V) {
	uint32 tmp = A;

	if (PPU_hook) PPU_hook(A);

	if (tmp < 0x2000) {
		if (PPUCHRRAM & (1 << (tmp >> 10)))
			VPage[tmp >> 10][tmp] = V;
	} else if (tmp < 0x3F00) {
		if (PPUNTARAM & (1 << ((tmp & 0xF00) >> 10)))
			vnapage[((tmp & 0xF00) >> 10)][tmp & 0x3FF] = V;
	} else {
		if (!(tmp & 3)) {
			if (!(tmp & 0xC))
				PALRAM[0x00] = PALRAM[0x04] = PALRAM[0x08] = PALRAM[0x0C] = V & 0x3F;
			else
				UPALRAM[((tmp & 0xC) >> 2) - 1] = V & 0x3F;
		} else
			PALRAM[tmp & 0x1F] = V & 0x3F;
	}
}

int PPU::GetCHRAddress(int A) {
	if (cdloggerVideoDataSize) {
		int result = &VPage[A >> 10][A] - CHRptr[0];
		if ((result >= 0) && (result < (int)cdloggerVideoDataSize))
			return result;
	} else
		if(A < 0x2000) return A;
	return -1;
}

uint8 FASTCALL PPU::Read_Default(uint32 A) {
	uint32 tmp = A;

	if (PPU_hook) PPU_hook(A);

	if (tmp < 0x2000) {
		return VPage[tmp >> 10][tmp];
	} else if (tmp < 0x3F00) {
		return vnapage[(tmp >> 10) & 0x3][tmp & 0x3FF];
	} else {
		uint8 ret;
		if (!(tmp & 3)) {
			if (!(tmp & 0xC))
				ret = READPAL(0x00);
			else
				ret = READUPAL(((tmp & 0xC) >> 2) - 1);
		} else
			ret = READPAL(tmp & 0x1F);
		return ret;
	}
}

void PPU::getScroll(int &xpos, int &ypos) {
	if (newppu) {
		ypos = ppur._vt * 8 + ppur._fv + ppur._v * 256;
		xpos = ppur._ht * 8 + ppur.fh + ppur._h * 256;
	} else {
		xpos = ((RefreshAddr & 0x400) >> 2) | ((RefreshAddr & 0x1F) << 3) | XOffset;

		ypos = ((RefreshAddr & 0x3E0) >> 2) | ((RefreshAddr & 0x7000) >> 12);
		if (RefreshAddr & 0x800) ypos += 240;
	}
}
//---------------

uint8 PPU::A2002(uint32 A) {
	if (newppu) {
		//once we thought we clear latches here, but that caused midframe glitches.
		//i think we should only reset the state machine for 2005/2006
		//ppur.clear_latches();
	}

	uint8 ret;

	LineUpdate();
	ret = Status();
	ret |= PPUGenLatch & 0x1F;

#ifdef FCEUDEF_DEBUGGER
	if (!fceuindbg)
#endif
	{
		vtoggle = 0;
		updateStatus(Status() & 0x7F);
		PPUGenLatch = ret;
	}

	return ret;
}

uint8 PPU::A2004(uint32 A) {
	if (newppu) {
		if ((ppur.status.sl < 241) && PPUON()) {
			// from cycles 0 to 63, the
			// 32 byte OAM buffer gets init
			// to 0xFF
			if (ppur.status.cycle < 64)
				return spr_read.ret = 0xFF;
			else {
				for (int i = spr_read.last;
					 i != ppur.status.cycle; ++i) {
					if (i < 256) {
						switch (spr_read.mode) {
						case 0:
							if (spr_read.count < 2)
								spr_read.ret = (data[3] & 0xF8) + (spr_read.count << 2);
							else
								spr_read.ret = spr_read.count << 2;

							spr_read.found_pos[spr_read.found] = spr_read.ret;
							spr_read.ret = SPRAM[spr_read.ret];

							if (i & 1) {
								//odd cycle
								//see if in range
								if (!((ppur.status.sl - 1 - spr_read.ret) & ~(Sprite16() ? 0xF : 0x7))) {
									++spr_read.found;
									spr_read.fetch = 1;
									spr_read.mode = 1;
								} else {
									if (++spr_read.count == 64) {
										spr_read.mode = 4;
										spr_read.count = 0;
									} else if (spr_read.found == 8) {
										spr_read.fetch = 0;
										spr_read.mode = 2;
									}
								}
							}
							break;
						case 1:	//sprite is in range fetch next 3 bytes
							if (i & 1) {
								++spr_read.fetch;
								if (spr_read.fetch == 4) {
									spr_read.fetch = 1;
									if (++spr_read.count == 64) {
										spr_read.count = 0;
										spr_read.mode = 4;
									} else if (spr_read.found == 8) {
										spr_read.fetch = 0;
										spr_read.mode = 2;
									} else
										spr_read.mode = 0;
								}
							}

							if (spr_read.count < 2)
								spr_read.ret = (data[3] & 0xF8) + (spr_read.count << 2);
							else
								spr_read.ret = spr_read.count << 2;

							spr_read.ret = SPRAM[spr_read.ret | spr_read.fetch];
							break;
						case 2:	//8th sprite fetched
							spr_read.ret = SPRAM[(spr_read.count << 2) | spr_read.fetch];
							if (i & 1) {
								if (!((ppur.status.sl - 1 - SPRAM[((spr_read.count << 2) | spr_read.fetch)]) & ~((Sprite16()) ? 0xF : 0x7))) {
									spr_read.fetch = 1;
									spr_read.mode = 3;
								} else {
									if (++spr_read.count == 64) {
										spr_read.count = 0;
										spr_read.mode = 4;
									}
									spr_read.fetch =
										(spr_read.fetch + 1) & 3;
								}
							}
							spr_read.ret = spr_read.count;
							break;
						case 3:	//9th sprite overflow detected
							spr_read.ret = SPRAM[spr_read.count | spr_read.fetch];
							if (i & 1) {
								if (++spr_read.fetch == 4) {
									spr_read.count = (spr_read.count + 1) & 63;
									spr_read.mode = 4;
								}
							}
							break;
						case 4:	//read OAM[n][0] until hblank
							if (i & 1)
								spr_read.count = (spr_read.count + 1) & 63;
							spr_read.fetch = 0;
							spr_read.ret = SPRAM[spr_read.count << 2];
							break;
						}
					} else if (i < 320) {
						spr_read.ret = (i & 0x38) >> 3;
						if (spr_read.found < (spr_read.ret + 1)) {
							if (spr_read.num) {
								spr_read.ret = SPRAM[252];
								spr_read.num = 0;
							} else
								spr_read.ret = 0xFF;
						} else if ((i & 7) < 4) {
							spr_read.ret =
								SPRAM[spr_read.found_pos[spr_read.ret] | spr_read.fetch++];
							if (spr_read.fetch == 4)
								spr_read.fetch = 0;
						} else
							spr_read.ret = SPRAM[spr_read.found_pos [spr_read.ret | 3]];
					} else {
						if (!spr_read.found)
							spr_read.ret = SPRAM[252];
						else
							spr_read.ret = SPRAM[spr_read.found_pos[0]];
						break;
					}
				}
				spr_read.last = ppur.status.cycle;
				return spr_read.ret;
			}
		} else
			return SPRAM[data[3]];
	} else {
		LineUpdate();
		return PPUGenLatch;
	}
}

uint8 PPU::A200x(uint32 A) {	/* Not correct for $2004 reads. */
	LineUpdate();
	return PPUGenLatch;
}

uint8 PPU::A2007(uint32 A) {
	uint8 ret;
	uint32 tmp = RefreshAddr & 0x3FFF;

	if (logger.debug_loggingCD) {
		if (!DummyRead && (LogAddress != -1)) {
			if (!(cdloggervdata[LogAddress] & 2)) {
				cdloggervdata[LogAddress] |= 2;
				if ((!(cdloggervdata[LogAddress] & 1)) && cdloggerVideoDataSize) undefinedvromcount--;
				vromreadcount++;
			}
		} else
			DummyRead = 0;
	}

	if (newppu) {
		ret = VRAMBuffer;
		RefreshAddr = ppur.get_2007access() & 0x3FFF;
		if ((RefreshAddr & 0x3F00) == 0x3F00) {
			//if it is in the palette range bypass the
			//delayed read, and what gets filled in the temp
			//buffer is the address - 0x1000, also
			//if grayscale is set then the return is AND with 0x30
			//to get a gray color reading
			if (!(tmp & 3)) {
				if (!(tmp & 0xC))
					ret = READPAL(0x00);
				else
					ret = READUPAL(((tmp & 0xC) >> 2) - 1);
			} else
				ret = READPAL(tmp & 0x1F);
			VRAMBuffer = Read(RefreshAddr - 0x1000);
		} else {
			if (logger.debug_loggingCD && (RefreshAddr < 0x2000))
				LogAddress = GetCHRAddress(RefreshAddr);
			VRAMBuffer = Read(RefreshAddr);
		}
		ppur.increment2007(ppur.status.sl >= 0 && ppur.status.sl < 241 && PPUON(), INC32() != 0);
		RefreshAddr = ppur.get_2007access();
		return ret;
	} else {

		//OLDPPU
		LineUpdate();

		if (tmp >= 0x3F00) {	// Palette RAM tied directly to the output data, without VRAM buffer
			if (!(tmp & 3)) {
				if (!(tmp & 0xC))
					ret = READPAL(0x00);
				else
					ret = READUPAL(((tmp & 0xC) >> 2) - 1);
			} else
				ret = READPAL(tmp & 0x1F);
			#ifdef FCEUDEF_DEBUGGER
			if (!fceuindbg)
			#endif
			{
				if ((tmp - 0x1000) < 0x2000)
					VRAMBuffer = VPage[(tmp - 0x1000) >> 10][tmp - 0x1000];
				else
					VRAMBuffer = vnapage[((tmp - 0x1000) >> 10) & 0x3][(tmp - 0x1000) & 0x3FF];
				if (PPU_hook) PPU_hook(tmp);
			}
		} else {
			ret = VRAMBuffer;
			#ifdef FCEUDEF_DEBUGGER
			if (!fceuindbg)
			#endif
			{
				if (PPU_hook) PPU_hook(tmp);
				PPUGenLatch = VRAMBuffer;
				if (tmp < 0x2000) {

					if (logger.debug_loggingCD)
						LogAddress = GetCHRAddress(tmp);
					if(MMC5Hack)
					{
						//probably wrong CD logging in this case...
						VRAMBuffer = *MMC5BGVRAMADR(tmp);
					}
					else VRAMBuffer = VPage[tmp >> 10][tmp];

				} else if (tmp < 0x3F00)
					VRAMBuffer = vnapage[(tmp >> 10) & 0x3][tmp & 0x3FF];
			}
		}

	#ifdef FCEUDEF_DEBUGGER
		if (!fceuindbg)
	#endif
		{
			if ((ScreenON() || SpriteON()) && (scanline < 240)) {
				uint32 rad = RefreshAddr;
				if ((rad & 0x7000) == 0x7000) {
					rad ^= 0x7000;
					if ((rad & 0x3E0) == 0x3A0)
						rad ^= 0xBA0;
					else if ((rad & 0x3E0) == 0x3e0)
						rad ^= 0x3e0;
					else
						rad += 0x20;
				} else
					rad += 0x1000;
				RefreshAddr = rad;
			} else {
				if (INC32())
					RefreshAddr += 32;
				else
					RefreshAddr++;
			}
			if (PPU_hook) PPU_hook(RefreshAddr & 0x3fff);
		}
		return ret;
	}
}

void PPU::B2000(uint32 A, uint8 V) {
	LineUpdate();
	PPUGenLatch = V;

	if (!(data[0] & 0x80) && (V & 0x80) && (Status() & 0x80))
		x6502->TriggerNMI2();

	data[0] = V;
	TempAddr &= 0xF3FF;
	TempAddr |= (V & 3) << 10;

	ppur._h = V & 1;
	ppur._v = (V >> 1) & 1;
	ppur.s = (V >> 4) & 1;
}

void PPU::B2001(uint32 A, uint8 V) {
	LineUpdate();
	if (paldeemphswap)
		V = (V&0x9F)|((V&0x40)>>1)|((V&0x20)<<1);
	PPUGenLatch = V;
	data[1] = V;
	if (V & 0xE0)
		deemp = V >> 5;
}

void PPU::B2002(uint32 A, uint8 V) {
	PPUGenLatch = V;
}

void PPU::B2003(uint32 A, uint8 V) {
	PPUGenLatch = V;
	data[3] = V;
	PPUSPL = V & 0x7;
}

void PPU::B2004(uint32 A, uint8 V) {
	PPUGenLatch = V;
	if (newppu) {
		//the attribute upper bits are not connected
		//so AND them out on write, since reading them
		//should return 0 in those bits.
		if ((data[3] & 3) == 2)
			V &= 0xE3;
		SPRAM[data[3]] = V;
		data[3] = (data[3] + 1) & 0xFF;
	} else {
		if (PPUSPL >= 8) {
			if (data[3] >= 8)
				SPRAM[data[3]] = V;
		} else {
			SPRAM[PPUSPL] = V;
		}
		data[3]++;
		PPUSPL++;
	}
}

void PPU::B2005(uint32 A, uint8 V) {
	uint32 tmp = TempAddr;
	LineUpdate();
	PPUGenLatch = V;
	if (!vtoggle) {
		tmp &= 0xFFE0;
		tmp |= V >> 3;
		XOffset = V & 7;
		ppur._ht = V >> 3;
		ppur.fh = V & 7;
	} else {
		tmp &= 0x8C1F;
		tmp |= ((V & ~0x7) << 2);
		tmp |= (V & 7) << 12;
		ppur._vt = V >> 3;
		ppur._fv = V & 7;
	}
	TempAddr = tmp;
	vtoggle ^= 1;
}


void PPU::B2006(uint32 A, uint8 V) {
	LineUpdate();

	PPUGenLatch = V;
	if (!vtoggle) {
		TempAddr &= 0x00FF;
		TempAddr |= (V & 0x3f) << 8;

		ppur._vt &= 0x07;
		ppur._vt |= (V & 0x3) << 3;
		ppur._h = (V >> 2) & 1;
		ppur._v = (V >> 3) & 1;
		ppur._fv = (V >> 4) & 3;
	} else {
		TempAddr &= 0xFF00;
		TempAddr |= V;

		RefreshAddr = TempAddr;
		DummyRead = 1;
		if (PPU_hook)
			PPU_hook(RefreshAddr);

		ppur._vt &= 0x18;
		ppur._vt |= (V >> 5);
		ppur._ht = V & 31;

		ppur.install_latches();
	}

	vtoggle ^= 1;
}

void PPU::B2007(uint32 A, uint8 V) {
	uint32 tmp = RefreshAddr & 0x3FFF;

	if (logger.debug_loggingCD) {
		if(!cdloggerVideoDataSize && (tmp < 0x2000))
			cdloggervdata[tmp] = 0;
	}

	if (newppu) {
		PPUGenLatch = V;
		RefreshAddr = ppur.get_2007access() & 0x3FFF;
		Write(RefreshAddr, V);
		ppur.increment2007(ppur.status.sl >= 0 && ppur.status.sl < 241 && PPUON(), INC32() != 0);
		RefreshAddr = ppur.get_2007access();
	} else {
		PPUGenLatch = V;
		if (tmp < 0x2000) {
			if (PPUCHRRAM & (1 << (tmp >> 10)))
				VPage[tmp >> 10][tmp] = V;
		} else if (tmp < 0x3F00) {
			if (PPUNTARAM & (1 << ((tmp & 0xF00) >> 10)))
				vnapage[((tmp & 0xF00) >> 10)][tmp & 0x3FF] = V;
		} else {
			if (!(tmp & 3)) {
				if (!(tmp & 0xC))
					PALRAM[0x00] = PALRAM[0x04] = PALRAM[0x08] = PALRAM[0x0C] = V & 0x3F;
				else
					UPALRAM[((tmp & 0xC) >> 2) - 1] = V & 0x3F;
			} else
				PALRAM[tmp & 0x1F] = V & 0x3F;
		}
		if (INC32())
			RefreshAddr += 32;
		else
			RefreshAddr++;
		if (PPU_hook)
			PPU_hook(RefreshAddr & 0x3fff);
	}
}

void PPU::B4014(uint32 A, uint8 V) {
	uint32 t = V << 8;
	int x;

	for (x = 0; x < 256; x++)
		x6502->DMW(0x2004, x6502->DMR(t + x));
	SpriteDMA = V;
}

void PPU::ResetRL(uint8 *target) {
	memset(target, 0xFF, 256);
	InputScanlineHook(0, 0, 0, 0);
	Plinef = target;
	Pline = target;
	firsttile = 0;
	linestartts = x6502->timestamp() * 48 + x6502->count();
	tofix = 0;
	LineUpdate();
	tofix = 1;
}

void PPU::LineUpdate(void) {
	if (newppu)
		return;

#ifdef FCEUDEF_DEBUGGER
	if (!fceuindbg)
#endif
	if (Pline) {
		int l = GETLASTPIXEL();
		RefreshLine(l);
	}
}

void PPU::SetRenderPlanes(bool sprites, bool bg) {
	rendersprites = sprites;
	renderbg = bg;
}

void PPU::GetRenderPlanes(bool& sprites, bool& bg) {
	sprites = rendersprites;
	bg = renderbg;
}

void PPU::EndRL(void) {
	RefreshLine(272);
	if (tofix)
		Fixit1();
	CheckSpriteHit(272);
	Pline = 0;
}

void PPU::CheckSpriteHit(int p) {
	int l = p - 16;
	int x;

	if (sphitx == 0x100) return;

	for (x = sphitx; x < (sphitx + 8) && x < l; x++) {
		if ((sphitdata & (0x80 >> (x - sphitx))) && !(Plinef[x] & 64) && x < 255) {
			updateStatus(Status() | 0x40);
			sphitx = 0x100;
			break;
		}
	}
}

// lasttile is really "second to last tile."
void PPU::RefreshLine(int lastpixel) {
	uint32 smorkus = RefreshAddr;

	#define RefreshAddr smorkus
	uint32 vofs;
	int X1;

	register uint8 *P = Pline;
	int lasttile = lastpixel >> 3;
	int numtiles;

	if (norecurse) return;

	if (sphitx != 0x100 && !(Status() & 0x40)) {
		if ((sphitx < (lastpixel - 16)) && !(sphitx < ((lasttile - 2) * 8)))
			lasttile++;
	}

	if (lasttile > 34) lasttile = 34;
	numtiles = lasttile - firsttile;

	if (numtiles <= 0) return;

	P = Pline;

	vofs = 0;

	if(PEC586Hack)
		vofs = ((RefreshAddr & 0x200) << 3) | ((RefreshAddr >> 12) & 7);
	else
		vofs = ((data[0] & 0x10) << 8) | ((RefreshAddr >> 12) & 7);

	if (!ScreenON() && !SpriteON()) {
		uint32 tem;
		tem = READPAL(0) | (READPAL(0) << 8) | (READPAL(0) << 16) | (READPAL(0) << 24);
		tem |= 0x40404040;
		FCEU_dwmemset(Pline, tem, numtiles * 8);
		P += numtiles * 8;
		Pline = P;

		firsttile = lasttile;

		#define TOFIXNUM (272 - 0x4)
		if (lastpixel >= TOFIXNUM && tofix) {
			Fixit1();
			tofix = 0;
		}

		if ((lastpixel - 16) >= 0) {
			InputScanlineHook(Plinef, spork ? sprlinebuf : 0, linestartts, lasttile * 8 - 16);
		}
		return;
	}

	//Priority bits, needed for sprite emulation.
	PALRAM[0] |= 64;
	PALRAM[4] |= 64;
	PALRAM[8] |= 64;
	PALRAM[0xC] |= 64;

	//This high-level graphics MMC5 emulation code was written for MMC5 carts in "CL" mode.
	//It's probably not totally correct for carts in "SL" mode.

#define PPUT_MMC5
	if (MMC5Hack && geniestage != 1) {
		if (MMC5HackCHRMode == 0 && (MMC5HackSPMode & 0x80)) {
			int tochange = MMC5HackSPMode & 0x1F;
			tochange -= firsttile;
			for (X1 = firsttile; X1 < lasttile; X1++) {
				if ((tochange <= 0 && MMC5HackSPMode & 0x40) || (tochange > 0 && !(MMC5HackSPMode & 0x40))) {
					#define PPUT_MMC5SP
					#include "pputile_obj.inc"
					#undef PPUT_MMC5SP
				} else {
					#include "pputile_obj.inc"
				}
				tochange--;
			}
		} else if (MMC5HackCHRMode == 1 && (MMC5HackSPMode & 0x80)) {
			int tochange = MMC5HackSPMode & 0x1F;
			tochange -= firsttile;

			#define PPUT_MMC5SP
			#define PPUT_MMC5CHR1
			for (X1 = firsttile; X1 < lasttile; X1++) {
				#include "pputile_obj.inc"
			}
			#undef PPUT_MMC5CHR1
			#undef PPUT_MMC5SP
		} else if (MMC5HackCHRMode == 1) {
			#define PPUT_MMC5CHR1
			for (X1 = firsttile; X1 < lasttile; X1++) {
				#include "pputile_obj.inc"
			}
			#undef PPUT_MMC5CHR1
		} else {
			for (X1 = firsttile; X1 < lasttile; X1++) {
				#include "pputile_obj.inc"
			}
		}
	}
	#undef PPUT_MMC5
	else if (PPU_hook) {
		norecurse = 1;
		#define PPUT_HOOK
		if (PEC586Hack) {
			#define PPU_BGFETCH
			for (X1 = firsttile; X1 < lasttile; X1++) {
				#include "pputile_obj.inc"
			}
			#undef PPU_BGFETCH
		} else {
			for (X1 = firsttile; X1 < lasttile; X1++) {
				#include "pputile_obj.inc"
			}
		}
		#undef PPUT_HOOK
		norecurse = 0;
	} else {
		if (PEC586Hack) {
			#define PPU_BGFETCH
			for (X1 = firsttile; X1 < lasttile; X1++) {
				#include "pputile_obj.inc"
			}
			#undef PPU_BGFETCH
		} else {
			for (X1 = firsttile; X1 < lasttile; X1++) {
				#include "pputile_obj.inc"
			}
		}
	}

#undef vofs
#undef RefreshAddr

	//Reverse changes made before.
	PALRAM[0] &= 63;
	PALRAM[4] &= 63;
	PALRAM[8] &= 63;
	PALRAM[0xC] &= 63;

	RefreshAddr = smorkus;
	if (firsttile <= 2 && 2 < lasttile && !(data[1] & 2)) {
		uint32 tem;
		tem = READPAL(0) | (READPAL(0) << 8) | (READPAL(0) << 16) | (READPAL(0) << 24);
		tem |= 0x40404040;
		*(uint32*)Plinef = *(uint32*)(Plinef + 4) = tem;
	}

	if (!ScreenON()) {
		uint32 tem;
		int tstart, tcount;
		tem = READPAL(0) | (READPAL(0) << 8) | (READPAL(0) << 16) | (READPAL(0) << 24);
		tem |= 0x40404040;

		tcount = lasttile - firsttile;
		tstart = firsttile - 2;
		if (tstart < 0) {
			tcount += tstart;
			tstart = 0;
		}
		if (tcount > 0)
			FCEU_dwmemset(Plinef + tstart * 8, tem, tcount * 8);
	}

	if (lastpixel >= TOFIXNUM && tofix) {
		Fixit1();
		tofix = 0;
	}

	//This only works right because of a hack earlier in this function.
	CheckSpriteHit(lastpixel);

	if ((lastpixel - 16) >= 0) {
		InputScanlineHook(Plinef, spork ? sprlinebuf : 0, linestartts, lasttile * 8 - 16);
	}
	Pline = P;
	firsttile = lasttile;
}

void PPU::Fixit1(void) {
	if (ScreenON() || SpriteON()) {
		uint32 rad = RefreshAddr;

		if ((rad & 0x7000) == 0x7000) {
			rad ^= 0x7000;
			if ((rad & 0x3E0) == 0x3A0)
				rad ^= 0xBA0;
			else if ((rad & 0x3E0) == 0x3e0)
				rad ^= 0x3e0;
			else
				rad += 0x20;
		} else
			rad += 0x1000;
		RefreshAddr = rad;
	}
}

INLINE void PPU::Fixit2(void) {
	if (ScreenON() || SpriteON()) {
		uint32 rad = RefreshAddr;
		rad &= 0xFBE0;
		rad |= TempAddr & 0x041f;
		RefreshAddr = rad;
	}
}

void MMC5_hb(int);		//Ugh ugh ugh.
void PPU::DoLine(void) {
	if (scanline >= 240 && scanline != totalscanlines) {
		x6502->Run(256 + 69);
		scanline++;
		x6502->Run(16);
		return;
	}

	int x;
	uint8 *target = XBuf + ((scanline < 240 ? scanline : 240) << 8);
	u8* dtarget = XDBuf + ((scanline < 240 ? scanline : 240) << 8);

	if (MMC5Hack) MMC5_hb(scanline);

	x6502->Run(256);
	EndRL();

	if (!renderbg) {// User asked to not display background data.
		uint32 tem;
		uint8 col;
		if (gNoBGFillColor == 0xFF)
			col = READPAL(0);
		else col = gNoBGFillColor;
		tem = col | (col << 8) | (col << 16) | (col << 24);
		tem |= 0x40404040; 
		FCEU_dwmemset(target, tem, 256);
	}

	if (SpriteON())
		CopySprites(target);

	//greyscale handling (mask some bits off the color) ? ? ?
	if (ScreenON() || SpriteON())
	{
		if (data[1] & 0x01) {
			for (x = 63; x >= 0; x--)
				*(uint32*)&target[x << 2] = (*(uint32*)&target[x << 2]) & 0x30303030;
		}
	}

	//some pathetic attempts at deemph
	if ((data[1] >> 5) == 0x7) {
		for (x = 63; x >= 0; x--)
			*(uint32*)&target[x << 2] = ((*(uint32*)&target[x << 2]) & 0x3f3f3f3f) | 0xc0c0c0c0;
	} else if (data[1] & 0xE0)
		for (x = 63; x >= 0; x--)
			*(uint32*)&target[x << 2] = (*(uint32*)&target[x << 2]) | 0x40404040;
	else
		for (x = 63; x >= 0; x--)
			*(uint32*)&target[x << 2] = ((*(uint32*)&target[x << 2]) & 0x3f3f3f3f) | 0x80808080;

	//write the actual deemph
	for (x = 63; x >= 0; x--)
		*(uint32*)&dtarget[x << 2] = ((data[1]>>5)<<0)|((data[1]>>5)<<8)|((data[1]>>5)<<16)|((data[1]>>5)<<24);

	sphitx = 0x100;

	if (ScreenON() || SpriteON())
		FetchSpriteData();

	if (GameHBIRQHook && (ScreenON() || SpriteON()) && ((data[0] & 0x38) != 0x18)) {
		x6502->Run(6);
		Fixit2();
		x6502->Run(4);
		GameHBIRQHook();
		x6502->Run(85 - 16 - 10);
	} else {
		x6502->Run(6);	// Tried 65, caused problems with Slalom(maybe others)
		Fixit2();
		x6502->Run(85 - 6 - 16);

		// A semi-hack for Star Trek: 25th Anniversary
		if (GameHBIRQHook && (ScreenON() || SpriteON()) && ((data[0] & 0x38) != 0x18))
			GameHBIRQHook();
	}

	DEBUG(FCEUD_UpdateNTView(scanline, 0));

	if (SpriteON())
		RefreshSprites();
	if (GameHBIRQHook2 && (ScreenON() || SpriteON()))
		GameHBIRQHook2();
	scanline++;
	if (scanline < 240) {
		ResetRL(XBuf + (scanline << 8));
	}
	x6502->Run(16);
}

void PPU::FetchSpriteData(void) {
	uint8 ns, sb;
	SPR *spr;
	uint8 H;
	int n;
	int vofs;
	uint8 P0 = data[0];

	spr = (SPR*)SPRAM;
	H = 8;

	ns = sb = 0;

	vofs = (uint32)(P0 & 0x8 & (((P0 & 0x20) ^ 0x20) >> 2)) << 9;
	H += (P0 & 0x20) >> 2;

	if (!PPU_hook)
		for (n = 63; n >= 0; n--, spr++) {
			if ((uint32)(scanline - spr->y) >= H) continue;
			if (ns < maxsprites) {
				if (n == 63) sb = 1;

				{
					SPRB dst;
					uint8 *C;
					int t;
					uint32 vadr;

					t = (int)scanline - (spr->y);

					if (Sprite16())
						vadr = ((spr->no & 1) << 12) + ((spr->no & 0xFE) << 4);
					else
						vadr = (spr->no << 4) + vofs;

					if (spr->atr & V_FLIP) {
						vadr += 7;
						vadr -= t;
						vadr += (P0 & 0x20) >> 1;
						vadr -= t & 8;
					} else {
						vadr += t;
						vadr += t & 8;
					}

					/* Fix this geniestage hack */
					if (MMC5Hack && geniestage != 1)
						C = MMC5SPRVRAMADR(vadr);
					else
						C = VRAMADR(vadr);

					if (SpriteON())
						RENDER_LOG(vadr);
					dst.ca[0] = C[0];
					if (SpriteON())
						RENDER_LOG(vadr + 8);
					dst.ca[1] = C[8];
					dst.x = spr->x;
					dst.atr = spr->atr;

					*(uint32*)&SPRBUF[ns << 2] = *(uint32*)&dst;
				}

				ns++;
			} else {
				updateStatus(Status() | 0x20);
				break;
			}
		}
	else
		for (n = 63; n >= 0; n--, spr++) {
			if ((uint32)(scanline - spr->y) >= H) continue;

			if (ns < maxsprites) {
				if (n == 63) sb = 1;

				{
					SPRB dst;
					uint8 *C;
					int t;
					uint32 vadr;

					t = (int)scanline - (spr->y);

					if (Sprite16())
						vadr = ((spr->no & 1) << 12) + ((spr->no & 0xFE) << 4);
					else
						vadr = (spr->no << 4) + vofs;

					if (spr->atr & V_FLIP) {
						vadr += 7;
						vadr -= t;
						vadr += (P0 & 0x20) >> 1;
						vadr -= t & 8;
					} else {
						vadr += t;
						vadr += t & 8;
					}

					if (MMC5Hack)
						C = MMC5SPRVRAMADR(vadr);
					else
						C = VRAMADR(vadr);
					if (SpriteON())
						RENDER_LOG(vadr);
					dst.ca[0] = C[0];
					if (ns < 8) {
						PPU_hook(0x2000);
						PPU_hook(vadr);
					}
					if (SpriteON())
						RENDER_LOG(vadr + 8);
					dst.ca[1] = C[8];
					dst.x = spr->x;
					dst.atr = spr->atr;


					*(uint32*)&SPRBUF[ns << 2] = *(uint32*)&dst;
				}

				ns++;
			} else {
				updateStatus(Status() | 0x20);
				break;
			}
		}

	//Handle case when >8 sprites per scanline option is enabled.
	if (ns > 8) updateStatus(Status() | 0x20);
	else if (PPU_hook) {
		for (n = 0; n < (8 - ns); n++) {
			PPU_hook(0x2000);
			PPU_hook(vofs);
		}
	}
	numsprites = ns;
	SpriteBlurp = sb;
}

void PPU::RefreshSprites(void) {
	int n;
	SPRB *spr;

	spork = 0;
	if (!numsprites) return;

	FCEU_dwmemset(sprlinebuf, 0x80808080, 256);
	numsprites--;
	spr = (SPRB*)SPRBUF + numsprites;

	for (n = numsprites; n >= 0; n--, spr--) {
		uint32 pixdata;
		uint8 J, atr;

		int x = spr->x;
		uint8 *C;
		int VB;

		pixdata = ppulut1[spr->ca[0]] | ppulut2[spr->ca[1]];
		J = spr->ca[0] | spr->ca[1];
		atr = spr->atr;

		if (J) {
			if (n == 0 && SpriteBlurp && !(Status() & 0x40)) {
				sphitx = x;
				sphitdata = J;
				if (atr & H_FLIP)
					sphitdata = ((J << 7) & 0x80) |
								((J << 5) & 0x40) |
								((J << 3) & 0x20) |
								((J << 1) & 0x10) |
								((J >> 1) & 0x08) |
								((J >> 3) & 0x04) |
								((J >> 5) & 0x02) |
								((J >> 7) & 0x01);
			}

			C = sprlinebuf + x;
			VB = (0x10) + ((atr & 3) << 2);

			if (atr & SP_BACK) {
				if (atr & H_FLIP) {
					if (J & 0x80) C[7] = READPAL(VB | (pixdata & 3)) | 0x40;
					pixdata >>= 4;
					if (J & 0x40) C[6] = READPAL(VB | (pixdata & 3)) | 0x40;
					pixdata >>= 4;
					if (J & 0x20) C[5] = READPAL(VB | (pixdata & 3)) | 0x40;
					pixdata >>= 4;
					if (J & 0x10) C[4] = READPAL(VB | (pixdata & 3)) | 0x40;
					pixdata >>= 4;
					if (J & 0x08) C[3] = READPAL(VB | (pixdata & 3)) | 0x40;
					pixdata >>= 4;
					if (J & 0x04) C[2] = READPAL(VB | (pixdata & 3)) | 0x40;
					pixdata >>= 4;
					if (J & 0x02) C[1] = READPAL(VB | (pixdata & 3)) | 0x40;
					pixdata >>= 4;
					if (J & 0x01) C[0] = READPAL(VB | pixdata) | 0x40;
				} else {
					if (J & 0x80) C[0] = READPAL(VB | (pixdata & 3)) | 0x40;
					pixdata >>= 4;
					if (J & 0x40) C[1] = READPAL(VB | (pixdata & 3)) | 0x40;
					pixdata >>= 4;
					if (J & 0x20) C[2] = READPAL(VB | (pixdata & 3)) | 0x40;
					pixdata >>= 4;
					if (J & 0x10) C[3] = READPAL(VB | (pixdata & 3)) | 0x40;
					pixdata >>= 4;
					if (J & 0x08) C[4] = READPAL(VB | (pixdata & 3)) | 0x40;
					pixdata >>= 4;
					if (J & 0x04) C[5] = READPAL(VB | (pixdata & 3)) | 0x40;
					pixdata >>= 4;
					if (J & 0x02) C[6] = READPAL(VB | (pixdata & 3)) | 0x40;
					pixdata >>= 4;
					if (J & 0x01) C[7] = READPAL(VB | pixdata) | 0x40;
				}
			} else {
				if (atr & H_FLIP) {
					if (J & 0x80) C[7] = READPAL(VB | (pixdata & 3));
					pixdata >>= 4;
					if (J & 0x40) C[6] = READPAL(VB | (pixdata & 3));
					pixdata >>= 4;
					if (J & 0x20) C[5] = READPAL(VB | (pixdata & 3));
					pixdata >>= 4;
					if (J & 0x10) C[4] = READPAL(VB | (pixdata & 3));
					pixdata >>= 4;
					if (J & 0x08) C[3] = READPAL(VB | (pixdata & 3));
					pixdata >>= 4;
					if (J & 0x04) C[2] = READPAL(VB | (pixdata & 3));
					pixdata >>= 4;
					if (J & 0x02) C[1] = READPAL(VB | (pixdata & 3));
					pixdata >>= 4;
					if (J & 0x01) C[0] = READPAL(VB | pixdata);
				} else {
					if (J & 0x80) C[0] = READPAL(VB | (pixdata & 3));
					pixdata >>= 4;
					if (J & 0x40) C[1] = READPAL(VB | (pixdata & 3));
					pixdata >>= 4;
					if (J & 0x20) C[2] = READPAL(VB | (pixdata & 3));
					pixdata >>= 4;
					if (J & 0x10) C[3] = READPAL(VB | (pixdata & 3));
					pixdata >>= 4;
					if (J & 0x08) C[4] = READPAL(VB | (pixdata & 3));
					pixdata >>= 4;
					if (J & 0x04) C[5] = READPAL(VB | (pixdata & 3));
					pixdata >>= 4;
					if (J & 0x02) C[6] = READPAL(VB | (pixdata & 3));
					pixdata >>= 4;
					if (J & 0x01) C[7] = READPAL(VB | pixdata);
				}
			}
		}
	}
	SpriteBlurp = 0;
	spork = 1;
}

void PPU::CopySprites(uint8 *target) {
	uint8 n = ((data[1] & 4) ^ 4) << 1;
	uint8 *P = target;

	if (!spork) return;
	spork = 0;

	if (!rendersprites) return;	//User asked to not display sprites.

 loopskie:
	{
		uint32 t = *(uint32*)(sprlinebuf + n);

		if (t != 0x80808080) {
			#ifdef LSB_FIRST
			if (!(t & 0x80)) {
				if (!(t & 0x40) || (P[n] & 0x40))		// Normal sprite || behind bg sprite
					P[n] = sprlinebuf[n];
			}

			if (!(t & 0x8000)) {
				if (!(t & 0x4000) || (P[n + 1] & 0x40))		// Normal sprite || behind bg sprite
					P[n + 1] = (sprlinebuf + 1)[n];
			}

			if (!(t & 0x800000)) {
				if (!(t & 0x400000) || (P[n + 2] & 0x40))	// Normal sprite || behind bg sprite
					P[n + 2] = (sprlinebuf + 2)[n];
			}

			if (!(t & 0x80000000)) {
				if (!(t & 0x40000000) || (P[n + 3] & 0x40))	// Normal sprite || behind bg sprite
					P[n + 3] = (sprlinebuf + 3)[n];
			}
			#else
			/* TODO:  Simplify */
			if (!(t & 0x80000000)) {
				if (!(t & 0x40000000))	// Normal sprite
					P[n] = sprlinebuf[n];
				else if (P[n] & 64)		// behind bg sprite
					P[n] = sprlinebuf[n];
			}

			if (!(t & 0x800000)) {
				if (!(t & 0x400000))	// Normal sprite
					P[n + 1] = (sprlinebuf + 1)[n];
				else if (P[n + 1] & 64)	// behind bg sprite
					P[n + 1] = (sprlinebuf + 1)[n];
			}

			if (!(t & 0x8000)) {
				if (!(t & 0x4000))		// Normal sprite
					P[n + 2] = (sprlinebuf + 2)[n];
				else if (P[n + 2] & 64)	// behind bg sprite
					P[n + 2] = (sprlinebuf + 2)[n];
			}

			if (!(t & 0x80)) {
				if (!(t & 0x40))		// Normal sprite
					P[n + 3] = (sprlinebuf + 3)[n];
				else if (P[n + 3] & 64)	// behind bg sprite
					P[n + 3] = (sprlinebuf + 3)[n];
			}
			#endif
		}
	}
	n += 4;
	if (n) goto loopskie;
}

void PPU::SetVideoSystem(int w) {
	if (w) {
		scanlines_per_frame = dendy ? 262: 312;
		FSettings->FirstSLine = FSettings->UsrFirstSLine[1];
		FSettings->LastSLine = FSettings->UsrLastSLine[1];
		//paldeemphswap = 1; // dendy has pal ppu, and pal ppu has these swapped
	} else {
		scanlines_per_frame = 262;
		FSettings->FirstSLine = FSettings->UsrFirstSLine[0];
		FSettings->LastSLine = FSettings->UsrLastSLine[0];
		//paldeemphswap = 0;
	}
}

void PPU::Reset(void) {
	VRAMBuffer = data[0] = data[1] = data[2] = data[3] = 0;
	PPUSPL = 0;
	PPUGenLatch = 0;
	RefreshAddr = TempAddr = 0;
	vtoggle = 0;
	ppudead = 2;
	kook = 0;
	idleSynch = 1;

	new_ppu_reset = true; // delay reset of ppur/spr_read until it's ready to start a new frame
}

void PPU::Power(void) {

	memset(NTARAM, 0x00, 0x800);
	memset(PALRAM, 0x00, 0x20);
	memset(UPALRAM, 0x00, 0x03);
	memset(SPRAM, 0x00, 0x100);
	Reset();

	handler->SetReadHandler(0x0000, 0xFFFF, &ANull_);
	handler->SetWriteHandler(0x0000, 0xFFFF, &BNull_);

	handler->SetReadHandler(0, 0x7FF, &ARAML_);
	handler->SetWriteHandler(0, 0x7FF, &BRAML_);

	handler->SetReadHandler(0x800, 0x1FFF, &ARAMH_);	// Part of a little
	handler->SetWriteHandler(0x800, 0x1FFF, &BRAMH_);	//hack for a small speed boost.

	for (int x = 0x2000; x < 0x4000; x += 8) {
		handler->SetReadHandler(x, &A200x_);
		handler->SetWriteHandler(x, &B2000_);
		handler->SetReadHandler(x + 1, &A200x_);
		handler->SetWriteHandler(x + 1, &B2001_);
		handler->SetReadHandler(x + 2, &A2002_);
		handler->SetWriteHandler(x + 2, &B2002_);
		handler->SetReadHandler(x + 3, &A200x_);
		handler->SetWriteHandler(x + 3, &B2003_);
		handler->SetReadHandler(x + 4, &A2004_);
		handler->SetWriteHandler(x + 4, &B2004_);
		handler->SetReadHandler(x + 5, &A200x_);
		handler->SetWriteHandler(x + 5, &B2005_);
		handler->SetReadHandler(x + 6, &A200x_);
		handler->SetWriteHandler(x + 6, &B2006_);
		handler->SetReadHandler(x + 7, &A2007_);
		handler->SetWriteHandler(x + 7, &B2007_);
	}
    handler->SetWriteHandler(0x4014, &B4014_);
}

int PPU::Loop(int skip) {
	if ((newppu) && (GameInfo->type != GIT_NSF)) {
		int NewLoop(int skip);
		return NewLoop(skip);
	}

	//Needed for Knight Rider, possibly others.
	if (ppudead) {
		memset(XBuf, 0x80, 256 * 240);
		x6502->Run(scanlines_per_frame * (256 + 85));
		ppudead--;
	} else {
		x6502->Run(256 + 85);
		updateStatus(Status() | 0x80);

		//Not sure if this is correct.  According to Matt Conte and my own tests, it is.
		//Timing is probably off, though.
		//NOTE:  Not having this here breaks a Super Donkey Kong game.
		data[3] = PPUSPL = 0;

		//I need to figure out the true nature and length of this delay.
		x6502->Run(12);
		if (GameInfo->type == GIT_NSF)
			DoNSFFrame();
		else {
			if (VBlankON())
				x6502->TriggerNMI();
		}
		x6502->Run((scanlines_per_frame - 242) * (256 + 85) - 12);
		if (overclock_enabled && vblankscanlines) {
			if (!DMC_7bit || !skip_7bit_overclocking) {
				overclocking = 1;
				x6502->Run(vblankscanlines * (256 + 85) - 12);
				overclocking = 0;
			}
		}
		updateStatus(Status() & 0x1f);
		x6502->Run(256);

		{
			int x;

			if (ScreenON() || SpriteON()) {
				if (GameHBIRQHook && ((data[0] & 0x38) != 0x18))
					GameHBIRQHook();
				if (PPU_hook)
					for (x = 0; x < 42; x++) {
						PPU_hook(0x2000); PPU_hook(0);
					}
				if (GameHBIRQHook2)
					GameHBIRQHook2();
			}
			x6502->Run(85 - 16);
			if (ScreenON() || SpriteON()) {
				RefreshAddr = TempAddr;
				if (PPU_hook) PPU_hook(RefreshAddr & 0x3fff);
			}

			//Clean this stuff up later.
			spork = numsprites = 0;
			ResetRL(XBuf);

			x6502->Run(16 - kook);
			kook ^= 1;
		}
		if (GameInfo->type == GIT_NSF)
			x6502->Run((256 + 85) * normalscanlines);
		#ifdef FRAMESKIP
		else if (skip) {
			int y;

			y = SPRAM[0];
			y++;

			updateStatus(Status() | 0x20);	// Fixes "Bee 52".  Does it break anything?
			if (GameHBIRQHook) {
				x6502->Run(256);
				for (scanline = 0; scanline < 240; scanline++) {
					if (ScreenON() || SpriteON())
						GameHBIRQHook();
					if (scanline == y && SpriteON()) updateStatus(Status() | 0x40);
					x6502->Run((scanline == 239) ? 85 : (256 + 85));
				}
			} else if (y < 240) {
				x6502->Run((256 + 85) * y);
				if (SpriteON()) updateStatus(Status() | 0x40);	// Quick and very dirty hack.
				x6502->Run((256 + 85) * (240 - y));
			} else
				x6502->Run((256 + 85) * 240);
		}
		#endif
		else {
			deemp = data[1] >> 5;

			// manual samples can't play correctly with overclocking
			if (DMC_7bit && skip_7bit_overclocking) // 7bit sample started before 240th line
				totalscanlines = normalscanlines;
			else
				totalscanlines = normalscanlines + (overclock_enabled ? postrenderscanlines : 0);

			for (scanline = 0; scanline < totalscanlines; ) {	//scanline is incremented in  DoLine.  Evil. :/
				deempcnt[deemp]++;
				if (scanline < 240)
					DEBUG(FCEUD_UpdatePPUView(scanline, 1));
				DoLine();

				if (scanline < normalscanlines || scanline == totalscanlines)
					overclocking = 0;
				else {
					if (DMC_7bit && skip_7bit_overclocking) // 7bit sample started after 240th line
						break;
					overclocking = 1;
				}
			}
			DMC_7bit = 0;

			if (MMC5Hack) MMC5_hb(scanline);

			//deemph nonsense, kept for complicated reasons (see SetNESDeemph_OldHacky implementation)
			int maxref = 0;
			for (int x = 1, max = 0; x < 7; x++) {
				if (deempcnt[x] > max) {
					max = deempcnt[x];
					maxref = x;
				}
				deempcnt[x] = 0;
			}
			SetNESDeemph_OldHacky(maxref, 0);
		}
	}	//else... to if(ppudead)

	#ifdef FRAMESKIP
	if (skip) {
		FCEU_PutImageDummy();
		return(0);
	} else
	#endif
	{
		FCEU_PutImage();
		return(1);
	}
}

void PPU::LoadState(int version) {
	TempAddr = TempAddrT;
	RefreshAddr = RefreshAddrT;
}

void PPU::SaveState(void) {
	TempAddrT = TempAddr;
	RefreshAddrT = RefreshAddr;
}

uint32 PPU::PeekAddress()
{
	if (newppu)
	{
		return ppur.get_2007access() & 0x3FFF;
	}

	return RefreshAddr & 0x3FFF;
}

void PPU::runppu(int x) {
	ppur.status.cycle = (ppur.status.cycle + x) % ppur.status.end_cycle;
	if (!new_ppu_reset) // if resetting, suspend CPU until the first frame
	{
		x6502->Run(x);
	}
}

inline int PPU::PaletteAdjustPixel(int pixel) {
	if ((data[1] >> 5) == 0x7)
		return (pixel & 0x3f) | 0xc0;
	else if (data[1] & 0xE0)
		return pixel | 0x40;
	else
		return (pixel & 0x3F) | 0x80;
}

int PPU::NewLoop(int skip) {

	if (new_ppu_reset) // first frame since reset, time to initialize
	{
		ppur.reset();
		spr_read.reset();
		new_ppu_reset = false;
	}

	//262 scanlines
	if (ppudead) {
		// not quite emulating all the NES power up behavior
		// since it is known that the NES ignores writes to some
		// register before around a full frame, but no games
		// should write to those regs during that time, it needs
		// to wait for vblank
		ppur.status.sl = 241;
		if (PAL)
			runppu(70 * kLineTime);
		else
			runppu(20 * kLineTime);
		ppur.status.sl = 0;
		runppu(242 * kLineTime);
		--ppudead;
        goto finish;
	}

    {

	  updateStatus(Status() | 0x80);
	  ppuphase = PPUPHASE_VBL;

	  //Not sure if this is correct.  According to Matt Conte and my own tests, it is.
	  //Timing is probably off, though.
	  //NOTE:  Not having this here breaks a Super Donkey Kong game.
	  data[3] = PPUSPL = 0;
	  const int delay = 20;	//fceu used 12 here but I couldnt get it to work in marble madness and pirates.

	  ppur.status.sl = 241;	//for sprite reads

	  //formerly: runppu(delay);
	  for(int dot=0;dot<delay;dot++)
	  	runppu(1);

	  if (VBlankON()) x6502->TriggerNMI();
	  int sltodo = PAL?70:20;
	  	
	  //formerly: runppu(20 * (kLineTime) - delay);
	  for(int S=0;S<sltodo;S++)
	  {
	  	for(int dot=(S==0?delay:0);dot<kLineTime;dot++)
	  		runppu(1);
	  	ppur.status.sl++;
	  }

	  //this seems to run just before the dummy scanline begins
	  updateStatus(0);
	  //this early out caused metroid to fail to boot. I am leaving it here as a reminder of what not to do
	  //if(!PPUON()) { runppu(kLineTime*242); goto finish; }

	  //There are 2 conditions that update all 5 PPU scroll counters with the
	  //contents of the latches adjacent to them. The first is after a write to
	  //2006/2. The second, is at the beginning of scanline 20, when the PPU starts
	  //rendering data for the first time in a frame (this update won't happen if
	  //all rendering is disabled via 2001.3 and 2001.4).

	  //if(PPUON())
	  //	ppur.install_latches();

	  //capture the initial xscroll
	  //int xscroll = ppur.fh;
	  //render 241/291 scanlines (1 dummy at beginning, dendy's 50 at the end)
	  //ignore overclocking!
	  for (int sl = 0; sl < normalscanlines; sl++) {
	  	spr_read.start_scanline();

	  	g_rasterpos = 0;
	  	ppur.status.sl = sl;

	  	linestartts = x6502->timestamp() * 48 + x6502->count(); // pixel timestamp for debugger

	  	const int yp = sl - 1;
	  	ppuphase = PPUPHASE_BG;

	  	if (sl != 0 && sl < 241) { // ignore the invisible
	  		DEBUG(FCEUD_UpdatePPUView(scanline = yp, 1));
	  		DEBUG(FCEUD_UpdateNTView(scanline = yp, 1));
	  	}

	  	//hack to fix SDF ship intro screen with split. is it right?
	  	//well, if we didnt do this, we'd be passing in a negative scanline, so that's a sign something is fishy..
	  	if(sl != 0)
	  		if (MMC5Hack) MMC5_hb(yp);


	  	//twiddle the oam buffers
	  	const int scanslot = oamslot ^ 1;
	  	const int renderslot = oamslot;
	  	oamslot ^= 1;

	  	oamcount = oamcounts[renderslot];

	  	//the main scanline rendering loop:
	  	//32 times, we will fetch a tile and then render 8 pixels.
	  	//two of those tiles were read in the last scanline.
	  	for (int xt = 0; xt < 32; xt++) {
	  		bgdata.main[xt + 2].Read();

	  		const uint8 blank = (gNoBGFillColor == 0xFF) ? READPAL(0) : gNoBGFillColor;

	  		//ok, we're also going to draw here.
	  		//unless we're on the first dummy scanline
	  		if (sl != 0 && sl < 241) { // cape at 240 for dendy, its PPU does nothing afterwards
	  			int xstart = xt << 3;
	  			oamcount = oamcounts[renderslot];
	  			uint8 * const target = XBuf + (yp << 8) + xstart;
	  			uint8 * const dtarget = XDBuf + (yp << 8) + xstart;
	  			uint8 *ptr = target;
	  			uint8 *dptr = dtarget;
	  			int rasterpos = xstart;

	  			//check all the conditions that can cause things to render in these 8px
	  			const bool renderspritenow = SpriteON() && (xt > 0 || SpriteLeft8());
	  			const bool renderbgnow = ScreenON() && (xt > 0 || BGLeft8());
	  			for (int xp = 0; xp < 8; xp++, rasterpos++, g_rasterpos++) {
	  				//bg pos is different from raster pos due to its offsetability.
	  				//so adjust for that here
	  				const int bgpos = rasterpos + ppur.fh;
	  				const int bgpx = bgpos & 7;
	  				const int bgtile = bgpos >> 3;

	  				uint8 pixel = 0;
	  				uint8 pixelcolor = blank;

	  				//according to qeed's doc, use palette 0 or $2006's value if it is & 0x3Fxx
	  				if (!ScreenON() && !SpriteON())
	  				{
	  					// if there's anything wrong with how we're doing this, someone please chime in
	  					int addr = ppur.get_2007access();
	  					if ((addr & 0x3F00) == 0x3F00)
	  					{
	  						pixel = addr & 0x1F;
	  						}
	  					pixelcolor = PALRAM[pixel];
	  				}

	  				//generate the BG data
	  				if (renderbgnow) {
	  					uint8* pt = bgdata.main[bgtile].pt;
	  					pixel = ((pt[0] >> (7 - bgpx)) & 1) | (((pt[1] >> (7 - bgpx)) & 1) << 1) | bgdata.main[bgtile].at;
	  				}
	  				if (renderbg)
	  					pixelcolor = READPAL(pixel);

	  				//look for a sprite to be drawn
	  				bool havepixel = false;
	  				for (int s = 0; s < oamcount; s++) {
	  						uint8* oam = oams[renderslot][s];
	  					int x = oam[3];
	  					if (rasterpos >= x && rasterpos < x + 8) {
	  						//build the pixel.
	  						//fetch the LSB of the patterns
	  						uint8 spixel = oam[4] & 1;
	  						spixel |= (oam[5] & 1) << 1;

	  						//shift down the patterns so the next pixel is in the LSB
	  						oam[4] >>= 1;
	  						oam[5] >>= 1;

	  						if (!renderspritenow) continue;

	  						//bail out if we already have a pixel from a higher priority sprite
	  						if (havepixel) continue;

	  						//transparent pixel bailout
	  						if (spixel == 0) continue;

	  						//spritehit:
	  						//1. is it sprite#0?
	  						//2. is the bg pixel nonzero?
	  						//then, it is spritehit.
	  						if (oam[6] == 0 && (pixel & 3) != 0 &&
	  							rasterpos < 255) {
	  							updateStatus(Status() | 0x40);
	  						}
	  						havepixel = true;

	  						//priority handling
	  						if (oam[2] & 0x20) {
	  							//behind background:
	  							if ((pixel & 3) != 0) continue;
	  						}

	  						//bring in the palette bits and palettize
	  						spixel |= (oam[2] & 3) << 2;

	  						if (rendersprites)
	  							pixelcolor = READPAL(0x10 + spixel);
	  					}
	  				}

	  				*ptr++ = PaletteAdjustPixel(pixelcolor);
	  				*dptr++= data[1]>>5; //grab deemph
	  			}
	  		}
	  	}

	  	//look for sprites (was supposed to run concurrent with bg rendering)
	  	oamcounts[scanslot] = 0;
	  	oamcount = 0;
	  	const int spriteHeight = Sprite16() ? 16 : 8;
	  	for (int i = 0; i < 64; i++) {
	  		oams[scanslot][oamcount][7] = 0;
	  		uint8* spr = SPRAM + i * 4;
	  		if (yp >= spr[0] && yp < spr[0] + spriteHeight) {
	  			//if we already have maxsprites, then this new one causes an overflow,
	  			//set the flag and bail out.
	  			if (oamcount >= 8 && PPUON()) {
	  				updateStatus(Status() | 0x20);
	  				if (maxsprites == 8)
	  					break;
	  			}

	  			//just copy some bytes into the internal sprite buffer
	  			for (int j = 0; j < 4; j++)
	  				oams[scanslot][oamcount][j] = spr[j];
	  			oams[scanslot][oamcount][7] = 1;

	  			//note that we stuff the oam index into [6].
	  			//i need to turn this into a struct so we can have fewer magic numbers
	  			oams[scanslot][oamcount][6] = (uint8)i;
	  			oamcount++;
	  		}
	  	}
	  	oamcounts[scanslot] = oamcount;

	  	//FV is clocked by the PPU's horizontal blanking impulse, and therefore will increment every scanline.
	  	//well, according to (which?) tests, maybe at the end of hblank.
	  	//but, according to what it took to get crystalis working, it is at the beginning of hblank.

	  	//this is done at cycle 251
	  	//rendering scanline, it doesn't need to be scanline 0,
	  	//because on the first scanline when the increment is 0, the vs_scroll is reloaded.
	  	//if(PPUON() && sl != 0)
	  	//	ppur.increment_vs();

	  	//todo - think about clearing oams to a predefined value to force deterministic behavior

	  	ppuphase = PPUPHASE_OBJ;

	  	//fetch sprite patterns
	  	for (int s = 0; s < maxsprites; s++) {
	  		//if we have hit our eight sprite pattern and we dont have any more sprites, then bail
	  		if (s == oamcount && s >= 8)
	  			break;

	  		//if this is a real sprite sprite, then it is not above the 8 sprite limit.
	  		//this is how we support the no 8 sprite limit feature.
	  		//not that at some point we may need a virtual Read which just peeks and doesnt increment any counters
	  		//this could be handy for the debugging tools also
	  		const bool realSprite = (s < 8);

	  		uint8* const oam = oams[scanslot][s];
	  		uint32 line = yp - oam[0];
	  		if (oam[2] & 0x80)	//vflip
	  			line = spriteHeight - line - 1;

	  		uint32 patternNumber = oam[1];
	  		uint32 patternAddress;

	  		//create deterministic dummy fetch pattern
	  		if (!oam[7]) {
	  			patternNumber = 0;
	  			line = 0;
	  		}

	  		//8x16 sprite handling:
	  		if (Sprite16()) {
	  			uint32 bank = (patternNumber & 1) << 12;
	  			patternNumber = patternNumber & ~1;
	  			patternNumber |= (line >> 3);
	  			patternAddress = (patternNumber << 4) | bank;
	  		} else {
	  			patternAddress = (patternNumber << 4) | (SpAdrHI() << 9);
	  		}

	  		//offset into the pattern for the current line.
	  		//tricky: tall sprites have already had lines>8 taken care of by getting a new pattern number above.
	  		//so we just need the line offset for the second pattern
	  		patternAddress += line & 7;

	  		//garbage nametable fetches
	  		int garbage_todo = 2;
	  		if (PPUON())
	  		{
	  			if (sl == 0 && ppur.status.cycle == 304)
	  			{
	  				runppu(1);
	  				if (PPUON()) ppur.install_latches();
	  				runppu(1);
	  				garbage_todo = 0;
	  			}
	  			if ((sl != 0 && sl < 241) && ppur.status.cycle == 256)
	  			{
	  				runppu(1);
	  				//at 257: 3d world runner is ugly if we do this at 256
	  				if (PPUON()) ppur.install_h_latches();
	  				runppu(1);
	  				garbage_todo = 0;
	  			}
	  		}
	  		if (realSprite) runppu(garbage_todo);

	  		//Dragon's Lair (Europe version mapper 4)
	  		//does not set SpriteON in the beginning but it does
	  		//set the bg on so if using the conditional SpriteON the MMC3 counter
	  		//the counter will never count and no IRQs will be fired so use PPUON()
	  		if (((data[0] & 0x38) != 0x18) && s == 2 && PPUON()) {
	  			//(The MMC3 scanline counter is based entirely on PPU A12, triggered on rising edges (after the line remains low for a sufficiently long period of time))
	  			//http://nesdevwiki.org/wiki/index.php/Nintendo_MMC3
	  			//test cases for timing: SMB3, Crystalis
	  			//crystalis requires deferring this til somewhere in sprite [1,3]
	  			//kirby requires deferring this til somewhere in sprite [2,5..
	  			//if (PPUON() && GameHBIRQHook) {
	  			if (GameHBIRQHook) {
	  				GameHBIRQHook();
	  			}
	  		}

	  		//blind attempt to replicate old ppu functionality
	  		if(s == 2 && PPUON())
	  		{
	  			if (GameHBIRQHook2) {
	  				GameHBIRQHook2();
	  			}
	  		}

	  		if (realSprite) runppu(kFetchTime);


	  		//pattern table fetches
	  		RefreshAddr = patternAddress;
	  		if (SpriteON())
	  			RENDER_LOG(RefreshAddr);
	  		oam[4] = Read(RefreshAddr);
	  		if (realSprite) runppu(kFetchTime);

	  		RefreshAddr += 8;
	  		if (SpriteON())
	  			RENDER_LOG(RefreshAddr);
	  		oam[5] = Read(RefreshAddr);
	  		if (realSprite) runppu(kFetchTime);

	  		//hflip
	  		if (!(oam[2] & 0x40)) {
	  			oam[4] = bitrevlut[oam[4]];
	  			oam[5] = bitrevlut[oam[5]];
	  		}
	  	}

	  	ppuphase = PPUPHASE_BG;

	  	//fetch BG: two tiles for next line
	  	for (int xt = 0; xt < 2; xt++)
	  		bgdata.main[xt].Read();

	  	//I'm unclear of the reason why this particular access to memory is made.
	  	//The nametable address that is accessed 2 times in a row here, is also the
	  	//same nametable address that points to the 3rd tile to be rendered on the
	  	//screen (or basically, the first nametable address that will be accessed when
	  	//the PPU is fetching background data on the next scanline).
	  	//(not implemented yet)
	  	runppu(kFetchTime);
	  	if (sl == 0) {
	  		if (idleSynch && PPUON() && !PAL)
	  			ppur.status.end_cycle = 340;
	  		else
	  			ppur.status.end_cycle = 341;
	  		idleSynch ^= 1;
	  	} else
	  		ppur.status.end_cycle = 341;
	  	runppu(kFetchTime);

	  	//After memory access 170, the PPU simply rests for 4 cycles (or the
	  	//equivelant of half a memory access cycle) before repeating the whole
	  	//pixel/scanline rendering process. If the scanline being rendered is the very
	  	//first one on every second frame, then this delay simply doesn't exist.
	  	if (ppur.status.end_cycle == 341)
	  		runppu(1);
	  }	//scanline loop

	  DMC_7bit = 0;

	  if (MMC5Hack) MMC5_hb(240);

	      //idle for one line
	      runppu(kLineTime);
	      framectr++;
      }

finish:
    FCEU_PutImage();

    return 0;
}

}
