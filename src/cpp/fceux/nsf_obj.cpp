#include "nsf_obj.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "fceu_obj.h"

namespace fceu {

static const int FIXED_EXWRAM_SIZE = 32768+8192;

//mbg 7/31/06 todo - no reason this couldnt be assembled on the fly from actual asm source code. thatd be less obscure.
//here it is disassembled, for reference
/*
00:8000:8D F4 3F  STA $3FF4 = #$00
00:8003:A2 FF     LDX #$FF
00:8005:9A        TXS
00:8006:AD F0 3F  LDA $3FF0 = #$00
00:8009:F0 09     BEQ $8014
00:800B:AD F1 3F  LDA $3FF1 = #$00
00:800E:AE F3 3F  LDX $3FF3 = #$00
00:8011:20 00 00  JSR $0000
00:8014:A9 00     LDA #$00
00:8016:AA        TAX
00:8017:A8        TAY
00:8018:20 00 00  JSR $0000
00:801B:8D F5 EF  STA $EFF5 = #$FF
00:801E:90 FE     BCC $801E
00:8020:8D F3 3F  STA $3FF3 = #$00
00:8023:18        CLC
00:8024:90 FE     BCC $8024
*/
static uint8 NSFROM[0x30+6]=
{
	/* 0x00 - NMI */
	0x8D,0xF4,0x3F,       /* Stop play routine NMIs. */
	0xA2,0xFF,0x9A,       /* Initialize the stack pointer. */
	0xAD,0xF0,0x3F,       /* See if we need to init. */
	0xF0,0x09,            /* If 0, go to play routine playing. */

	0xAD,0xF1,0x3F,       /* Confirm and load A      */
	0xAE,0xF3,0x3F,       /* Load X with PAL/NTSC byte */

	0x20,0x00,0x00,       /* JSR to init routine     */

	0xA9,0x00,
	0xAA,
	0xA8,
	0x20,0x00,0x00,       /* JSR to play routine  */
	0x8D,0xF5,0x3F,        /* Start play routine NMIs. */
	0x90,0xFE,             /* Loopie time. */

	/* 0x20 */
	0x8D,0xF3,0x3F,        /* Init init NMIs */
	0x18,
	0x90,0xFE        /* Loopie time. */
};

uint8 NSF::NSFROMRead(uint32 A)
{
	return (NSFROM-0x3800)[A];
}

void NSF::NSFGI(GI h)
{
	switch(h)
	{
	case GI_CLOSE:
		if(NSFDATA) {free(NSFDATA);NSFDATA=0;}
		if(ExWRAM) {free(ExWRAM);ExWRAM=0;}
		if(NSFHeader.SoundChip&1) {
			//   NSFVRC6_Init();
		} else if(NSFHeader.SoundChip&2) {
			//   NSFVRC7_Init();
		} else if(NSFHeader.SoundChip&4) {
			//   FDSSoundReset();
		} else if(NSFHeader.SoundChip&8) {
			NSFMMC5_Close();
		} else if(NSFHeader.SoundChip&0x10) {
			//   NSFN106_Init();
		} else if(NSFHeader.SoundChip&0x20) {
			//   NSFAY_Init();
		}
		break;
	case GI_RESETM2:
	case GI_POWER: NSF_init();break;
	}
}

// First 32KB is reserved for sound chip emulation in the iNES mapper code.

inline void NSF::BANKSET(uint32 A, uint32 bank)
{
	bank&=NSFMaxBank;
	if(NSFHeader.SoundChip&4)
		memcpy(ExWRAM+(A-0x6000),NSFDATA+(bank<<12),4096);
	else
		fceu->cart.setprg4(A,bank);
}

int NSF::NSFLoad(const char *name, FCEUFILE *fp)
{
	int x;

    fceu::fseek(fp,0,SEEK_SET);
    fceu::fread(&NSFHeader,1,0x80,fp);
	if(memcmp(NSFHeader.ID,"NESM\x1a",5))
		return 0;
	NSFHeader.SongName[31]=NSFHeader.Artist[31]=NSFHeader.Copyright[31]=0;

	LoadAddr=NSFHeader.LoadAddressLow;
	LoadAddr|=NSFHeader.LoadAddressHigh<<8;

	if(LoadAddr<0x6000)
	{
        fceu::PrintError("Invalid load address.");
		return(0);
	}
	InitAddr=NSFHeader.InitAddressLow;
	InitAddr|=NSFHeader.InitAddressHigh<<8;

	PlayAddr=NSFHeader.PlayAddressLow;
	PlayAddr|=NSFHeader.PlayAddressHigh<<8;

	NSFSize=fceu::fgetsize(fp)-0x80;

	NSFMaxBank=((NSFSize+(LoadAddr&0xfff)+4095)/4096);
	NSFMaxBank=fceu->cart.PRGsize[0]=uppow2(NSFMaxBank);

	if(!(NSFDATA=(uint8 *)fceu::malloc(NSFMaxBank*4096)))
		return 0;

    fceu::fseek(fp,0x80,SEEK_SET);
	memset(NSFDATA,0x00,NSFMaxBank*4096);
    fceu::fread(NSFDATA+(LoadAddr&0xfff),1,NSFSize,fp);

	NSFMaxBank--;

	BSon=0;
	for(x=0;x<8;x++)
	{
		BSon|=NSFHeader.BankSwitch[x];
	}

	if(BSon==0)
	{
		BankCounter=0x00;

 		if ((NSFHeader.LoadAddressHigh & 0x70) >= 0x70)
		{
			//Ice Climber, and other F000 base address tunes need this
			BSon=0xFF;
		}
		else {
			for(x=(NSFHeader.LoadAddressHigh & 0x70) / 0x10;x<8;x++)
			{
				NSFHeader.BankSwitch[x]=BankCounter;
				BankCounter+=0x01;
			}
			BSon=0;
			}
	}

	for(x=0;x<8;x++)
		BSon|=NSFHeader.BankSwitch[x];

	fceu->GameInfo->type=GIT_NSF;
	fceu->GameInfo->input[0]=fceu->GameInfo->input[1]=SI_GAMEPAD;
	fceu->GameInfo->cspecial=SIS_NSF;

	for(x=0;;x++)
	{
		if(NSFROM[x]==0x20)
		{
			NSFROM[x+1]=InitAddr&0xFF;
			NSFROM[x+2]=InitAddr>>8;
			NSFROM[x+8]=PlayAddr&0xFF;
			NSFROM[x+9]=PlayAddr>>8;
			break;
		}
	}

	if(NSFHeader.VideoSystem==0)
		fceu->GameInfo->vidsys=GIV_NTSC;
	else if(NSFHeader.VideoSystem==1)
		fceu->GameInfo->vidsys=GIV_PAL;

	fceu->GameInterface=&NSFGI_;

	strcpy(fceu->LoadedRomFName,name);

    fceu::printf("\nNSF Loaded.\nFile information:\n");
    fceu::printf(" Name:       %s\n Artist:     %s\n Copyright:  %s\n\n",NSFHeader.SongName,NSFHeader.Artist,NSFHeader.Copyright);
	if(NSFHeader.SoundChip)
	{
		static char *tab[6]={"Konami VRCVI","Konami VRCVII","Nintendo FDS","Nintendo MMC5","Namco 106","Sunsoft FME-07"};

		for(x=0;x<6;x++)
			if(NSFHeader.SoundChip&(1<<x))
			{
                fceu::printf(" Expansion hardware:  %s\n",tab[x]);
				NSFHeader.SoundChip=1<<x;  /* Prevent confusing weirdness if more than one bit is set. */
				break;
			}
	}
	if(BSon)
		fceu::printf(" Bank-switched.\n");
    fceu::printf(" Load address:  $%04x\n Init address:  $%04x\n Play address:  $%04x\n",LoadAddr,InitAddr,PlayAddr);
    fceu::printf(" %s\n",(NSFHeader.VideoSystem&1)?"PAL":"NTSC");
    fceu::printf(" Starting song:  %d / %d\n\n",NSFHeader.StartingSong,NSFHeader.TotalSongs);

	//choose exwram size and allocate
	int exwram_size = 8192;
	if(NSFHeader.SoundChip&4)
		exwram_size = 32768+8192;
	//lets just always use this size, for savestate simplicity
	exwram_size = FIXED_EXWRAM_SIZE;
	ExWRAM=(uint8*)fceu::gmalloc(exwram_size);

	fceu->SetVidSystem(NSFHeader.VideoSystem);

	return 1;
}

uint8 NSF::NSFVectorRead(uint32 A)
{
	if(((NSFNMIFlags&1) && SongReload) || (NSFNMIFlags&2) || doreset)
	{
		if(A==0xFFFA) return(0x00);
		else if(A==0xFFFB) return(0x38);
		else if(A==0xFFFC) return(0x20);
		else if(A==0xFFFD) {doreset=0;return(0x38);}
		return(fceu->x6502.DB());
	}
	else
		return(fceu->cart.CartBR(A));
}

void NSF::NSF_init(void)
{
	doreset=1;

	fceu->cart.ResetCartMapping();
	if(NSFHeader.SoundChip&4)
	{
		fceu->cart.SetupCartPRGMapping(0,ExWRAM,32768+8192,1);
		fceu->cart.setprg32(0x6000,0);
		fceu->cart.setprg8(0xE000,4);
		memset(ExWRAM,0x00,32768+8192);
		fceu->handler.SetWriteHandler(0x6000,0xDFFF,&fceu->cart.CartBW_);
		fceu->handler.SetReadHandler(0x6000,0xFFFF,&fceu->cart.CartBR_);
	}
	else
	{
		memset(ExWRAM,0x00,8192);
		fceu->handler.SetReadHandler(0x6000,0x7FFF,&fceu->cart.CartBR_);
		fceu->handler.SetWriteHandler(0x6000,0x7FFF,&fceu->cart.CartBW_);
		fceu->cart.SetupCartPRGMapping(0,NSFDATA,((NSFMaxBank+1)*4096),0);
		fceu->cart.SetupCartPRGMapping(1,ExWRAM,8192,1);
		fceu->cart.setprg8r(1,0x6000,0);
		fceu->handler.SetReadHandler(0x8000,0xFFFF,&fceu->cart.CartBR_);
	}

	if(BSon)
	{
		int32 x;
		for(x=0;x<8;x++)
		{
			if(NSFHeader.SoundChip&4 && x>=6)
				BANKSET(0x6000+(x-6)*4096,NSFHeader.BankSwitch[x]);
			BANKSET(0x8000+x*4096,NSFHeader.BankSwitch[x]);
		}
	}
	else
	{
		int32 x;
		for(x=(LoadAddr&0xF000);x<0x10000;x+=0x1000)
			BANKSET(x,((x-(LoadAddr&0x7000))>>12));
	}

	fceu->handler.SetReadHandler(0xFFFA,0xFFFD,&NSFVectorRead_);

	fceu->handler.SetWriteHandler(0x2000,0x3fff,0);
	fceu->handler.SetReadHandler(0x2000,0x37ff,0);
	fceu->handler.SetReadHandler(0x3836,0x3FFF,0);
	fceu->handler.SetReadHandler(0x3800,0x3835,&NSFROMRead_);

	fceu->handler.SetWriteHandler(0x5ff6,0x5fff,&NSF_write_);

	fceu->handler.SetWriteHandler(0x3ff0,0x3fff,&NSF_write_);
	fceu->handler.SetReadHandler(0x3ff0,0x3fff,&NSF_read_);


	if(NSFHeader.SoundChip&1) {
		NSFVRC6_Init();
	} else if(NSFHeader.SoundChip&2) {
		NSFVRC7_Init();
	} else if(NSFHeader.SoundChip&4) {
		fceu->fds.FDSSoundReset();
	} else if(NSFHeader.SoundChip&8) {
		NSFMMC5_Init();
	} else if(NSFHeader.SoundChip&0x10) {
		NSFN106_Init();
	} else if(NSFHeader.SoundChip&0x20) {
		NSFAY_Init();
	}
	CurrentSong=NSFHeader.StartingSong;
	SongReload=0xFF;
	NSFNMIFlags=0;

	//zero 17-apr-2013 - added
	AddExState(StateRegs, ~0, 0, 0);
	AddExState(ExWRAM, FIXED_EXWRAM_SIZE, 0, "ERAM");
}

void NSF::NSF_write(uint32 A, uint8 V)
{
	switch(A)
	{
	case 0x3FF3:NSFNMIFlags|=1;break;
	case 0x3FF4:NSFNMIFlags&=~2;break;
	case 0x3FF5:NSFNMIFlags|=2;break;

	case 0x5FF6:
	case 0x5FF7:if(!(NSFHeader.SoundChip&4)) return;
	case 0x5FF8:
	case 0x5FF9:
	case 0x5FFA:
	case 0x5FFB:
	case 0x5FFC:
	case 0x5FFD:
	case 0x5FFE:
	case 0x5FFF:if(!BSon) return;
		A&=0xF;
		BANKSET((A*4096),V);
		break;
	}
}

uint8 NSF::NSF_read(uint32 A)
{
	int x;

	switch(A)
	{
	case 0x3ff0:x=SongReload;
		if(!fceu->ppu.get_fceuindbg())
			SongReload=0;
		return x;
	case 0x3ff1:
		if(!fceu->ppu.get_fceuindbg())
		{
			memset(fceu->RAM,0x00,0x800);

			(*fceu->handler.BWrite[0x4015])(0x4015,0x0);
			for(x=0;x<0x14;x++)
				(*fceu->handler.BWrite[0x4000+x])(0x4000+x,0);
			(*fceu->handler.BWrite[0x4015])(0x4015,0xF);

			if(NSFHeader.SoundChip&4)
			{
				(*fceu->handler.BWrite[0x4017])(0x4017,0xC0);  /* FDS BIOS writes $C0 */
				(*fceu->handler.BWrite[0x4089])(0x4089,0x80);
				(*fceu->handler.BWrite[0x408A])(0x408A,0xE8);
			}
			else
			{
				memset(ExWRAM,0x00,8192);
				(*fceu->handler.BWrite[0x4017])(0x4017,0xC0);
				(*fceu->handler.BWrite[0x4017])(0x4017,0xC0);
				(*fceu->handler.BWrite[0x4017])(0x4017,0x40);
			}

			if(BSon)
			{
				for(x=0;x<8;x++)
					BANKSET(0x8000+x*4096,NSFHeader.BankSwitch[x]);
			}
			#ifdef _S9XLUA_H
			//CallRegisteredLuaMemHook(A, 1, V, LUAMEMHOOK_WRITE); FIXME
			#endif
			return (CurrentSong-1);
		}
	case 0x3FF3:return fceu->PAL;
	}
	return 0;
}

void NSF::DrawNSF(uint8 *XBuf)
{
	char snbuf[16];
	int x;

	if(vismode==0) return;

	memset(XBuf,0,256*240);
	memset(XDBuf,0,256*240);


	{
		int32 *Bufpl;
		int32 mul=0;

		int l;
		l=GetSoundBuffer(&Bufpl);

		if(special==0)
		{
			if(fceu->FSettings.SoundVolume)
				mul=8192*240/(16384*fceu->FSettings.SoundVolume/50);
			for(x=0;x<256;x++)
			{
				uint32 y;
				y=142+((Bufpl[(x*l)>>8]*mul)>>14);
				if(y<240)
					XBuf[x+y*256]=3;
			}
		}
		else if(special==1)
		{
			if(fceu->FSettings.SoundVolume)
				mul=8192*240/(8192*fceu->FSettings.SoundVolume/50);
			for(x=0;x<256;x++)
			{
				double r;
				uint32 xp,yp;

				r=(Bufpl[(x*l)>>8]*mul)>>14;
				xp=128+r*cos(x*M_PI*2/256);
				yp=120+r*sin(x*M_PI*2/256);
				xp&=255;
				yp%=240;
				XBuf[xp+yp*256]=3;
			}
		}
		else if(special==2)
		{
			static double theta=0;
			if(fceu->FSettings.SoundVolume)
				mul=8192*240/(16384*fceu->FSettings.SoundVolume/50);
			for(x=0;x<128;x++)
			{
				double xc,yc;
				double r,t;
				uint32 m,n;

				xc=(double)128-x;
				yc=0-((double)( ((Bufpl[(x*l)>>8]) *mul)>>14));
				t=M_PI+atan(yc/xc);
				r=sqrt(xc*xc+yc*yc);

				t+=theta;
				m=128+r*cos(t);
				n=120+r*sin(t);

				if(m<256 && n<240)
					XBuf[m+n*256]=3;

			}
			for(x=128;x<256;x++)
			{
				double xc,yc;
				double r,t;
				uint32 m,n;

				xc=(double)x-128;
				yc=(double)((Bufpl[(x*l)>>8]*mul)>>14);
				t=atan(yc/xc);
				r=sqrt(xc*xc+yc*yc);

				t+=theta;
				m=128+r*cos(t);
				n=120+r*sin(t);

				if(m<256 && n<240)
					XBuf[m+n*256]=3;

			}
			theta+=(double)M_PI/256;
		}
	}

	static const int kFgColor = 1;
	fceu->drawing.DrawTextTrans(ClipSidesOffset+XBuf+10*256+4+(((31-strlen((char*)NSFHeader.SongName))<<2)), 256, NSFHeader.SongName, kFgColor);
	fceu->drawing.DrawTextTrans(ClipSidesOffset+XBuf+26*256+4+(((31-strlen((char*)NSFHeader.Artist))<<2)), 256,NSFHeader.Artist, kFgColor);
	fceu->drawing.DrawTextTrans(ClipSidesOffset+XBuf+42*256+4+(((31-strlen((char*)NSFHeader.Copyright))<<2)), 256,NSFHeader.Copyright, kFgColor);

	fceu->drawing.DrawTextTrans(ClipSidesOffset+XBuf+70*256+4+(((31-strlen("Song:"))<<2)), 256, (uint8*)"Song:", kFgColor);
	sprintf(snbuf,"<%d/%d>",CurrentSong,NSFHeader.TotalSongs);
	fceu->drawing.DrawTextTrans(XBuf+82*256+4+(((31-strlen(snbuf))<<2)), 256, (uint8*)snbuf, kFgColor);

	{
		uint8 tmp;
		tmp=fceu->input.GetJoyJoy();
		if((tmp&JOY_RIGHT) && !(last&JOY_RIGHT))
		{
			if(CurrentSong<NSFHeader.TotalSongs)
			{
				CurrentSong++;
				SongReload=0xFF;
			}
		}
		else if((tmp&JOY_LEFT) && !(last&JOY_LEFT))
		{
			if(CurrentSong>1)
			{
				CurrentSong--;
				SongReload=0xFF;
			}
		}
		else if((tmp&JOY_UP) && !(last&JOY_UP))
		{
			CurrentSong+=10;
			if(CurrentSong>NSFHeader.TotalSongs) CurrentSong=NSFHeader.TotalSongs;
			SongReload=0xFF;
		}
		else if((tmp&JOY_DOWN) && !(last&JOY_DOWN))
		{
			CurrentSong-=10;
			if(CurrentSong<1) CurrentSong=1;
			SongReload=0xFF;
		}
		else if((tmp&JOY_START) && !(last&JOY_START))
			SongReload=0xFF;
		else if((tmp&JOY_A) && !(last&JOY_A))
		{
			special=(special+1)%3;
		}
		last=tmp;
	}
}

void NSF::DoNSFFrame(void)
{
	if(((NSFNMIFlags&1) && SongReload) || (NSFNMIFlags&2))
		fceu->x6502.TriggerNMI();
}

void NSF::NSFSetVis(int mode)
{
	vismode=mode;
}

int NSF::NSFChange(int amount)
{
	CurrentSong+=amount;
	if(CurrentSong<1) CurrentSong=1;
	else if(CurrentSong>NSFHeader.TotalSongs) CurrentSong=NSFHeader.TotalSongs;
	SongReload=0xFF;

	return(CurrentSong);
}

//Returns total songs
int NSF::NSFGetInfo(uint8 *name, uint8 *artist, uint8 *copyright, int maxlen)
{
	strncpy((char*)name,(char*)NSFHeader.SongName,maxlen); //mbg merge 7/17/06 added casts
	strncpy((char*)artist,(char*)NSFHeader.Artist,maxlen); //mbg merge 7/17/06 added casts
	strncpy((char*)copyright,(char*)NSFHeader.Copyright,maxlen); //mbg merge 7/17/06 added casts
	return(NSFHeader.TotalSongs);
}

} // namespace fceu
