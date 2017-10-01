#include "sound_obj.h"

#include "fceu_obj.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>

namespace fceu {

static const int RectDuties[4]={1,2,4,6};

static const uint8 lengthtable[0x20]=
{
	10,254, 20,  2, 40,  4, 80,  6, 160,  8, 60, 10, 14, 12, 26, 14,
	12, 16, 24, 18, 48, 20, 96, 22, 192, 24, 72, 26, 16, 28, 32, 30
};


extern const uint32 NoiseFreqTableNTSC[0x10] =
{
	4, 8, 16, 32, 64, 96, 128, 160, 202,
	254, 380, 508, 762, 1016, 2034, 4068
};

extern const uint32 NoiseFreqTablePAL[0x10] =
{
	4, 7, 14, 30, 60, 88, 118, 148, 188,
	236, 354, 472, 708,  944, 1890, 3778
};


static const uint32 NTSCDMCTable[0x10]=
{
 428,380,340,320,286,254,226,214,
 190,160,142,128,106, 84 ,72,54
};

static const uint32 PALDMCTable[0x10]=
{
	398, 354, 316, 298, 276, 236, 210, 198,
	176, 148, 132, 118,  98,  78,  66,  50
};

void Sound::LoadDMCPeriod(uint8 V)
{
 if(fceu->PAL)
  DMCPeriod=PALDMCTable[V];
 else
  DMCPeriod=NTSCDMCTable[V];
}

void Sound::PrepDPCM()
{
 DMCAddress=0x4000+(DMCAddressLatch<<6);
 DMCSize=(DMCSizeLatch<<4)+1;
}

void Sound::LogDPCM(int romaddress, int dpcmsize) {
	int i = GetPRGAddress(romaddress);

	if(i == -1)return;

	for (int dpcmstart = i; dpcmstart < (i + dpcmsize); dpcmstart++) {
		if(!(cdloggerdata[dpcmstart] & 0x40)) {
			cdloggerdata[dpcmstart] |= 0x40;

			if(!(cdloggerdata[dpcmstart] & 2)){
				datacount++;
				cdloggerdata[dpcmstart] |= 2;
				if(!(cdloggerdata[dpcmstart] & 1))undefinedcount--;
			}
		}
	}
}

/* Instantaneous?  Maybe the new freq value is being calculated all of the time... */

int Sound::CheckFreq(uint32 cf, uint8 sr)
{
 uint32 mod;
 if(!(sr&0x8))
 {
  mod=cf>>(sr&7);
  if((mod+cf)&0x800)
   return(0);
 }
 return(1);
}

void Sound::SQReload(int x, uint8 V)
{
           if(EnabledChannels&(1<<x))
           {
            if(x)
             (*DoSQ2)();
            else
             (*DoSQ1)();
            lengthcount[x]=lengthtable[(V>>3)&0x1f];
	   }

           sweepon[x]=PSG[(x<<2)|1]&0x80;
           curfreq[x]=PSG[(x<<2)|0x2]|((V&7)<<8);
           SweepCount[x]=((PSG[(x<<2)|0x1]>>4)&7)+1;

           RectDutyCount[x]=7;
	   EnvUnits[x].reloaddec=1;
	   //reloadfreq[x]=1;
}

void Sound::Write_PSG(uint32 A, uint8 V)
{
	A&=0x1F;
	switch(A)
	{
	case 0x0:
		(*DoSQ1)();
		EnvUnits[0].Mode=(V&0x30)>>4;
		EnvUnits[0].Speed=(V&0xF);
		if (fceu->swapDuty)
			V = (V&0x3F)|((V&0x80)>>1)|((V&0x40)<<1);
		break;
	case 0x1:
		sweepon[0]=V&0x80;
		break;
	case 0x2:
		(*DoSQ1)();
		curfreq[0]&=0xFF00;
		curfreq[0]|=V;
		break;
	case 0x3:
		SQReload(0,V);
		break;
	case 0x4:
		(*DoSQ2)();
		EnvUnits[1].Mode=(V&0x30)>>4;
		EnvUnits[1].Speed=(V&0xF);
		if (fceu->swapDuty)
			V = (V&0x3F)|((V&0x80)>>1)|((V&0x40)<<1);
		break;
	case 0x5:
		sweepon[1]=V&0x80;
		break;
	case 0x6:
		(*DoSQ2)();
		curfreq[1]&=0xFF00;
		curfreq[1]|=V;
		break;
	case 0x7:
		SQReload(1,V);
		break;
	case 0xa:
		(*DoTriangle)();
		break;
	case 0xb:
		(*DoTriangle)();
		if(EnabledChannels&0x4)
			lengthcount[2]=lengthtable[(V>>3)&0x1f];
		TriMode=1;	// Load mode
		break;
	case 0xC:
		(*DoNoise)();
		EnvUnits[2].Mode=(V&0x30)>>4;
		EnvUnits[2].Speed=(V&0xF);
		break;
	case 0xE:
		(*DoNoise)();
		break;
	case 0xF:
		(*DoNoise)();
		if(EnabledChannels&0x8)
			lengthcount[3]=lengthtable[(V>>3)&0x1f];
		EnvUnits[2].reloaddec=1;
		break;
	case 0x10:
		(*DoPCM)();
		LoadDMCPeriod(V&0xF);
		if(SIRQStat&0x80)
		{
			if(!(V&0x80))
			{
				fceu->x6502.IRQEnd(FCEU_IQDPCM);
				SIRQStat&=~0x80;
			}
			else fceu->x6502.IRQBegin(FCEU_IQDPCM);
		}
		break;
	}
	PSG[A]=V;
}

void Sound::Write_DMCRegs(uint32 A, uint8 V)
{
	A&=0xF;
	
	switch(A)
	{
	case 0x00:
		(*DoPCM)();
	    LoadDMCPeriod(V&0xF);
	
	    if(SIRQStat&0x80)
	    {
			if(!(V&0x80))
			{
				fceu->x6502.IRQEnd(FCEU_IQDPCM);
				SIRQStat&=~0x80;
			}
			else fceu->x6502.IRQBegin(FCEU_IQDPCM);
	    }
		DMCFormat=V;
		break;
	case 0x01:
		(*DoPCM)();
		InitialRawDALatch=V&0x7F;
		RawDALatch=InitialRawDALatch;
		if (RawDALatch)
			DMC_7bit = 1;
		break;
	case 0x02:
		DMCAddressLatch=V;
		if (V)
			DMC_7bit = 0;
		break;
	case 0x03:
		DMCSizeLatch=V;
		if (V)
			DMC_7bit = 0;
		break;
	}
}

void Sound::StatusWrite(uint32 A, uint8 V)
{
	int x;

    (*DoSQ1)();
    (*DoSQ2)();
    (*DoTriangle)();
    (*DoNoise)();
    (*DoPCM)();

    for(x=0;x<4;x++)
		if(!(V&(1<<x))) lengthcount[x]=0;   /* Force length counters to 0. */

    if(V&0x10)
    {
		if(!DMCSize)
			PrepDPCM();
    }
	else
	{
		DMCSize=0;
	}
	SIRQStat&=~0x80;
	fceu->x6502.IRQEnd(FCEU_IQDPCM);
	EnabledChannels=V&0x1F;
}

uint8 Sound::StatusRead(uint32 A)
{
   int x;
   uint8 ret;

   ret=SIRQStat;

   for(x=0;x<4;x++) ret|=lengthcount[x]?(1<<x):0;
   if(DMCSize) ret|=0x10;

   #ifdef FCEUDEF_DEBUGGER
   if(!fceuindbg)
   #endif
   {
    SIRQStat&=~0x40;
    fceu->x6502.IRQEnd(FCEU_IQFCOUNT);
   }
   return ret;
}

void Sound::FrameSoundStuff(int V)
{
 int P;

 (*DoSQ1)();
 (*DoSQ2)();
 (*DoNoise)();
 (*DoTriangle)();

 if(!(V&1)) /* Envelope decay, linear counter, length counter, freq sweep */
 {
  if(!(PSG[8]&0x80))
   if(lengthcount[2]>0)
    lengthcount[2]--;

  if(!(PSG[0xC]&0x20))  /* Make sure loop flag is not set. */
   if(lengthcount[3]>0)
    lengthcount[3]--;

  for(P=0;P<2;P++)
  {
   if(!(PSG[P<<2]&0x20))  /* Make sure loop flag is not set. */
    if(lengthcount[P]>0)
     lengthcount[P]--;

   /* Frequency Sweep Code Here */
   /* xxxx 0000 */
   /* xxxx = hz.  120/(x+1)*/
   if(sweepon[P])
   {
    int32 mod=0;

    if(SweepCount[P]>0) SweepCount[P]--;
    if(SweepCount[P]<=0)
    {
     SweepCount[P]=((PSG[(P<<2)+0x1]>>4)&7)+1; //+1;
     if(PSG[(P<<2)+0x1]&0x8)
     {
      mod-=(P^1)+((curfreq[P])>>(PSG[(P<<2)+0x1]&7));
      if(curfreq[P] && (PSG[(P<<2)+0x1]&7)/* && sweepon[P]&0x80*/)
      {
       curfreq[P]+=mod;
      }
     }
     else
     {
      mod=curfreq[P]>>(PSG[(P<<2)+0x1]&7);
      if((mod+curfreq[P])&0x800)
      {
       sweepon[P]=0;
       curfreq[P]=0;
      }
      else
      {
       if(curfreq[P] && (PSG[(P<<2)+0x1]&7)/* && sweepon[P]&0x80*/)
       {
        curfreq[P]+=mod;
       }
      }
     }
    }
   }
   else  /* Sweeping is disabled: */
   {
    //curfreq[P]&=0xFF00;
    //curfreq[P]|=PSG[(P<<2)|0x2]; //|((PSG[(P<<2)|3]&7)<<8);
   }
  }
 }

 /* Now do envelope decay + linear counter. */

  if(TriMode) // In load mode?
   TriCount=PSG[0x8]&0x7F;
  else if(TriCount)
   TriCount--;

  if(!(PSG[0x8]&0x80))
   TriMode=0;

  for(P=0;P<3;P++)
  {
   if(EnvUnits[P].reloaddec)
   {
    EnvUnits[P].decvolume=0xF;
    EnvUnits[P].DecCountTo1=EnvUnits[P].Speed+1;
    EnvUnits[P].reloaddec=0;
    continue;
   }

   if(EnvUnits[P].DecCountTo1>0) EnvUnits[P].DecCountTo1--;
   if(EnvUnits[P].DecCountTo1==0)
   {
    EnvUnits[P].DecCountTo1=EnvUnits[P].Speed+1;
    if(EnvUnits[P].decvolume || (EnvUnits[P].Mode&0x2))
    {
     EnvUnits[P].decvolume--;
     EnvUnits[P].decvolume&=0xF;
    }
   }
  }
}

void Sound::FrameSoundUpdate(void)
{
 // Linear counter:  Bit 0-6 of $4008
 // Length counter:  Bit 4-7 of $4003, $4007, $400b, $400f

 if(!fcnt && !(IRQFrameMode&0x3))
 {
         SIRQStat|=0x40;
         fceu->x6502.IRQBegin(FCEU_IQFCOUNT);
 }

 if(fcnt==3)
 {
	if(IRQFrameMode&0x2)
	 fhcnt+=fhinc;
 }
 FrameSoundStuff(fcnt);
 fcnt=(fcnt+1)&3;
}


inline void Sound::tester(void)
{
 if(DMCBitCount==0)
 {
  if(!DMCHaveDMA)
   DMCHaveSample=0;
  else
  {
   DMCHaveSample=1;
   DMCShift=DMCDMABuf;
   DMCHaveDMA=0;
  }
 }
}

inline void Sound::DMCDMA(void)
{
  if(DMCSize && !DMCHaveDMA)
  {
   fceu->x6502.DMR(0x8000+DMCAddress);
   fceu->x6502.DMR(0x8000+DMCAddress);
   fceu->x6502.DMR(0x8000+DMCAddress);
   DMCDMABuf=fceu->x6502.DMR(0x8000+DMCAddress);
   DMCHaveDMA=1;
   DMCAddress=(DMCAddress+1)&0x7fff;
   DMCSize--;
   if(!DMCSize)
   {
    if(DMCFormat&0x40)
     PrepDPCM();
    else
    {
     SIRQStat|=0x80;
     if(DMCFormat&0x80)
      fceu->x6502.IRQBegin(FCEU_IQDPCM);
    }
   }
 }
}

void Sound::SoundCPUHook(int cycles)
{
 fhcnt-=cycles*48;
 if(fhcnt<=0)
 {
  FrameSoundUpdate();
  fhcnt+=fhinc;
 }

 DMCDMA();
 DMCacc-=cycles;

 while(DMCacc<=0)
 {
  if(DMCHaveSample)
  {
   uint8 bah=RawDALatch;
   int t=((DMCShift&1)<<2)-2;

   /* Unbelievably ugly hack */
   if(fceu->FSettings.SndRate)
   {
    soundtsoffs+=DMCacc;
    (*DoPCM)();
    soundtsoffs-=DMCacc;
   }
   RawDALatch+=t;
   if(RawDALatch&0x80)
    RawDALatch=bah;
  }

  DMCacc+=DMCPeriod;
  DMCBitCount=(DMCBitCount+1)&7;
  DMCShift>>=1;
  tester();
 }
}

void Sound::RDoPCM(void)
{
 uint32 V; //mbg merge 7/17/06 made uint32

 for(V=ChannelBC[4];V<SOUNDTS;V++)
  WaveHi[V]+=(((RawDALatch<<16)/256) * fceu->FSettings.PCMVolume)&(~0xFFFF); // TODO get rid of floating calculations to binary. set log volume scaling.
 ChannelBC[4]=SOUNDTS;
}

/* This has the correct phase.  Don't mess with it. */
inline void Sound::RDoSQ(int x)		//Int x decides if this is Square Wave 1 or 2
{
   int32 V;
   int32 amp, ampx;
   int32 rthresh;
   int32 *D;
   int32 currdc;
   int32 cf;
   int32 rc;

   if(curfreq[x]<8 || curfreq[x]>0x7ff)
    goto endit;
   if(!CheckFreq(curfreq[x],PSG[(x<<2)|0x1]))
    goto endit;
   if(!lengthcount[x])
    goto endit;

   if(EnvUnits[x].Mode&0x1)
    amp=EnvUnits[x].Speed;
   else
    amp=EnvUnits[x].decvolume;	//Set the volume of the Square Wave

   //Modify Square wave volume based on channel volume modifiers
   //adelikat: Note: the formulat x = x * y /100 does not yield exact results, but is "close enough" and avoids the need for using double vales or implicit cohersion which are slower (we need speed here)
   ampx = x ? fceu->FSettings.Square2Volume : fceu->FSettings.Square1Volume; // TODO OPTIMIZE ME!
   if (ampx != 256) amp = (amp * ampx) / 256; // CaH4e3: fixed - setting up maximum volume for square2 caused complete mute square2 channel

   amp<<=24;

   rthresh=RectDuties[(PSG[(x<<2)]&0xC0)>>6];

   D=&WaveHi[ChannelBC[x]];
   V=SOUNDTS-ChannelBC[x];

   currdc=RectDutyCount[x];
   cf=(curfreq[x]+1)*2;
   rc=wlcount[x];

   while(V>0)
   {
    if(currdc<rthresh)
     *D+=amp;
    rc--;
    if(!rc)
    {
     rc=cf;
     currdc=(currdc+1)&7;
    }
    V--;
    D++;
   }

   RectDutyCount[x]=currdc;
   wlcount[x]=rc;

   endit:
   ChannelBC[x]=SOUNDTS;
}

void Sound::RDoSQ1(void)
{
 RDoSQ(0);
}

void Sound::RDoSQ2(void)
{
 RDoSQ(1);
}

void Sound::RDoSQLQ(void)
{
   int32 start,end;
   int32 V;
   int32 amp[2], ampx;
   int32 rthresh[2];
   int32 freq[2];
   int x;
   int32 inie[2];

   int32 ttable[2][8];
   int32 totalout;

   start=ChannelBC[0];
   end=(SOUNDTS<<16)/soundtsinc;
   if(end<=start) return;
   ChannelBC[0]=end;

   for(x=0;x<2;x++)
   {
    int y;

    inie[x]=nesincsize;
    if(curfreq[x]<8 || curfreq[x]>0x7ff)
     inie[x]=0;
    if(!CheckFreq(curfreq[x],PSG[(x<<2)|0x1]))
     inie[x]=0;
    if(!lengthcount[x])
     inie[x]=0;

    if(EnvUnits[x].Mode&0x1)
     amp[x]=EnvUnits[x].Speed;
    else
     amp[x]=EnvUnits[x].decvolume;

	//Modify Square wave volume based on channel volume modifiers
	//adelikat: Note: the formulat x = x * y /100 does not yield exact results, but is "close enough" and avoids the need for using double vales or implicit cohersion which are slower (we need speed here)
    ampx = x ? fceu->FSettings.Square1Volume : fceu->FSettings.Square2Volume;  // TODO OPTIMIZE ME!
    if (ampx != 256) amp[x] = (amp[x] * ampx) / 256; // CaH4e3: fixed - setting up maximum volume for square2 caused complete mute square2 channel

    if(!inie[x]) amp[x]=0;    /* Correct? Buzzing in MM2, others otherwise... */

    rthresh[x]=RectDuties[(PSG[x*4]&0xC0)>>6];

    for(y=0;y<8;y++)
    {
     if(y < rthresh[x])
      ttable[x][y] = amp[x];
     else
      ttable[x][y] = 0;
    }
    freq[x]=(curfreq[x]+1)<<1;
    freq[x]<<=17;
   }

   totalout = wlookup1[ ttable[0][RectDutyCount[0]] + ttable[1][RectDutyCount[1]] ];

   if(!inie[0] && !inie[1])
   {
    for(V=start;V<end;V++)
     Wave[V>>4]+=totalout;
   }
   else
   for(V=start;V<end;V++)
   {
    //int tmpamp=0;
    //if(RectDutyCount[0]<rthresh[0])
    // tmpamp=amp[0];
    //if(RectDutyCount[1]<rthresh[1])
    // tmpamp+=amp[1];
    //tmpamp=wlookup1[tmpamp];
    //tmpamp = wlookup1[ ttable[0][RectDutyCount[0]] + ttable[1][RectDutyCount[1]] ];

    Wave[V>>4]+=totalout; //tmpamp;

    sqacc[0]-=inie[0];
    sqacc[1]-=inie[1];

    if(sqacc[0]<=0)
    {
     rea:
     sqacc[0]+=freq[0];
     RectDutyCount[0]=(RectDutyCount[0]+1)&7;
     if(sqacc[0]<=0) goto rea;
     totalout = wlookup1[ ttable[0][RectDutyCount[0]] + ttable[1][RectDutyCount[1]] ];
    }

    if(sqacc[1]<=0)
    {
     rea2:
     sqacc[1]+=freq[1];
     RectDutyCount[1]=(RectDutyCount[1]+1)&7;
     if(sqacc[1]<=0) goto rea2;
     totalout = wlookup1[ ttable[0][RectDutyCount[0]] + ttable[1][RectDutyCount[1]] ];
    }
   }
}

void Sound::RDoTriangle(void)
{
 uint32 V; //mbg merge 7/17/06 made uitn32
 int32 tcout;

 tcout=(tristep&0xF);
 if(!(tristep&0x10)) tcout^=0xF;
 tcout=(tcout*3) << 16;  //(tcout<<1);

 if(!lengthcount[2] || !TriCount)
 {           /* Counter is halted, but we still need to output. */
  /*int32 *start = &WaveHi[ChannelBC[2]];
  int32 count = SOUNDTS - ChannelBC[2];
  while(count--)
  {
   //Modify volume based on channel volume modifiers
   *start += (tcout/256*fceu->FSettings.TriangleVolume)&(~0xFFFF);  // TODO OPTIMIZE ME NOW DAMMIT!
   start++;
  }*/
  int32 cout = (tcout/256*fceu->FSettings.TriangleVolume)&(~0xFFFF);
  for(V=ChannelBC[2];V<SOUNDTS;V++)
   WaveHi[V]+=cout;
 }
 else
  for(V=ChannelBC[2];V<SOUNDTS;V++)
  {
    //Modify volume based on channel volume modifiers
	WaveHi[V]+=(tcout/256*fceu->FSettings.TriangleVolume)&(~0xFFFF);  // TODO OPTIMIZE ME!
    wlcount[2]--;
    if(!wlcount[2])
    {
     wlcount[2]=(PSG[0xa]|((PSG[0xb]&7)<<8))+1;
     tristep++;
     tcout=(tristep&0xF);
     if(!(tristep&0x10)) tcout^=0xF;
     tcout=(tcout*3) << 16;
    }
  }

 ChannelBC[2]=SOUNDTS;
}

void Sound::RDoTriangleNoisePCMLQ(void)
{
   static uint32 tcout=0;
   static int32 triacc=0;
   static int32 noiseacc=0;

   int32 V;
   int32 start,end;
   int32 freq[2];
   int32 inie[2];
   uint32 amptab[2];
   uint32 noiseout;
   int nshift;

   int32 totalout;

   start=ChannelBC[2];
   end=(SOUNDTS<<16)/soundtsinc;
   if(end<=start) return;
   ChannelBC[2]=end;

   inie[0]=inie[1]=nesincsize;

   freq[0]=(((PSG[0xa]|((PSG[0xb]&7)<<8))+1));

   if(!lengthcount[2] || !TriCount || freq[0]<=4)
    inie[0]=0;

   freq[0]<<=17;
   if(EnvUnits[2].Mode&0x1)
    amptab[0]=EnvUnits[2].Speed;
   else
    amptab[0]=EnvUnits[2].decvolume;

   //Modify Square wave volume based on channel volume modifiers
   //adelikat: Note: the formulat x = x * y /100 does not yield exact results, but is "close enough" and avoids the need for using double vales or implicit cohersion which are slower (we need speed here)
   if (fceu->FSettings.TriangleVolume != 256) amptab[0] = (amptab[0] * fceu->FSettings.TriangleVolume) / 256;  // TODO OPTIMIZE ME!

   amptab[1]=0;
   amptab[0]<<=1;

   if(!lengthcount[3])
    amptab[0]=inie[1]=0;  /* Quick hack speedup, set inie[1] to 0 */

   noiseout=amptab[(nreg>>0xe)&1];

   if(PSG[0xE]&0x80)
    nshift=8;
   else
    nshift=13;


   totalout = wlookup2[tcout+noiseout+RawDALatch];

   if(inie[0] && inie[1])
   {
    for(V=start;V<end;V++)
    {
     Wave[V>>4]+=totalout;

    triacc-=inie[0];
    noiseacc-=inie[1];

    if(triacc<=0)
    {
     rea:
     triacc+=freq[0]; //t;
     tristep=(tristep+1)&0x1F;
     if(triacc<=0) goto rea;
     tcout=(tristep&0xF);
     if(!(tristep&0x10)) tcout^=0xF;
     tcout=tcout*3;
      totalout = wlookup2[tcout+noiseout+RawDALatch];
    }

    if(noiseacc<=0)
    {
     rea2:
        //used to added <<(16+2) when the noise table
        //values were half.
     if(fceu->PAL)
       noiseacc+=NoiseFreqTablePAL[PSG[0xE]&0xF]<<(16+1);
 	 else
       noiseacc+=NoiseFreqTableNTSC[PSG[0xE]&0xF]<<(16+1);
     nreg=(nreg<<1)+(((nreg>>nshift)^(nreg>>14))&1);
     nreg&=0x7fff;
     noiseout=amptab[(nreg>>0xe)&1];
     if(noiseacc<=0) goto rea2;
      totalout = wlookup2[tcout+noiseout+RawDALatch];
    } /* noiseacc<=0 */
  } /* for(V=... */
}
  else if(inie[0])
  {
    for(V=start;V<end;V++)
    {
     Wave[V>>4]+=totalout;

     triacc-=inie[0];

     if(triacc<=0)
     {
      area:
      triacc+=freq[0]; //t;
      tristep=(tristep+1)&0x1F;
      if(triacc<=0) goto area;
      tcout=(tristep&0xF);
      if(!(tristep&0x10)) tcout^=0xF;
      tcout=tcout*3;
      totalout = wlookup2[tcout+noiseout+RawDALatch];
     }
    }
  }
  else if(inie[1])
  {
    for(V=start;V<end;V++)
    {
     Wave[V>>4]+=totalout;
     noiseacc-=inie[1];
     if(noiseacc<=0)
     {
      area2:
         //used to be added <<(16+2) when the noise table
         //values were half.
      if(fceu->PAL)
        noiseacc+=NoiseFreqTablePAL[PSG[0xE]&0xF]<<(16+1);
	  else
        noiseacc+=NoiseFreqTableNTSC[PSG[0xE]&0xF]<<(16+1);
      nreg=(nreg<<1)+(((nreg>>nshift)^(nreg>>14))&1);
      nreg&=0x7fff;
      noiseout=amptab[(nreg>>0xe)&1];
      if(noiseacc<=0) goto area2;
      totalout = wlookup2[tcout+noiseout+RawDALatch];
     } /* noiseacc<=0 */
    }
  }
  else
  {
    for(V=start;V<end;V++)
     Wave[V>>4]+=totalout;
  }
}

void Sound::RDoNoise(void)
{
 uint32 V; //mbg merge 7/17/06 made uint32
 int32 outo;
 uint32 amptab[2];

 if(EnvUnits[2].Mode&0x1)
  amptab[0]=EnvUnits[2].Speed;
 else
  amptab[0]=EnvUnits[2].decvolume;

 //Modfiy Noise channel volume based on channel volume setting
 //adelikat: Note: the formulat x = x * y /100 does not yield exact results, but is "close enough" and avoids the need for using double vales or implicit cohersion which are slower (we need speed here)
 if (fceu->FSettings.NoiseVolume != 256) amptab[0] = (amptab[0] * fceu->FSettings.NoiseVolume) / 256;  // TODO OPTIMIZE ME!
 amptab[0]<<=16;
 amptab[1]=0;

 amptab[0]<<=1;

 outo=amptab[(nreg>>0xe)&1];

 if(!lengthcount[3])
 {
  outo=amptab[0]=0;
 }

 if(PSG[0xE]&0x80)  // "short" noise
  for(V=ChannelBC[3];V<SOUNDTS;V++)
  {
   WaveHi[V]+=outo;
   wlcount[3]--;
   if(!wlcount[3])
   {
    uint8 feedback;
    if(fceu->PAL)
      wlcount[3]=NoiseFreqTablePAL[PSG[0xE]&0xF];
	else
      wlcount[3]=NoiseFreqTableNTSC[PSG[0xE]&0xF];
    feedback=((nreg>>8)&1)^((nreg>>14)&1);
    nreg=(nreg<<1)+feedback;
    nreg&=0x7fff;
    outo=amptab[(nreg>>0xe)&1];
   }
  }
 else
  for(V=ChannelBC[3];V<SOUNDTS;V++)
  {
   WaveHi[V]+=outo;
   wlcount[3]--;
   if(!wlcount[3])
   {
    uint8 feedback;
    if(fceu->PAL)
      wlcount[3]=NoiseFreqTablePAL[PSG[0xE]&0xF];
	else
      wlcount[3]=NoiseFreqTableNTSC[PSG[0xE]&0xF];
    feedback=((nreg>>13)&1)^((nreg>>14)&1);
    nreg=(nreg<<1)+feedback;
    nreg&=0x7fff;
    outo=amptab[(nreg>>0xe)&1];
   }
  }
 ChannelBC[3]=SOUNDTS;
}

void Sound::Write_IRQFM(uint32 A, uint8 V)
{
 V=(V&0xC0)>>6;
 fcnt=0;
 if(V&0x2)
  FrameSoundUpdate();
 fcnt=1;
 fhcnt=fhinc;
 fceu->x6502.IRQEnd(FCEU_IQFCOUNT);
 SIRQStat&=~0x40;
 IRQFrameMode=V;
}

void Sound::SetNESSoundMap(void)
{
  fceu->handler.SetWriteHandler(0x4000,0x400F,&Write_PSG_);
  fceu->handler.SetWriteHandler(0x4010,0x4013,&Write_DMCRegs_);
  fceu->handler.SetWriteHandler(0x4017,0x4017,&Write_IRQFM_);

  fceu->handler.SetWriteHandler(0x4015,0x4015,&StatusWrite_);
  fceu->handler.SetReadHandler(0x4015,0x4015,&StatusRead_);
}

int Sound::FlushEmulateSound(void)
{
  int x;
  int32 end,left;

  if(!fceu->x6502.soundtimestamp) return(0);

  if(!fceu->FSettings.SndRate)
  {
   left=0;
   end=0;
   goto nosoundo;
  }

  (*DoSQ1)();
  (*DoSQ2)();
  (*DoTriangle)();
  (*DoNoise)();
  (*DoPCM)();

  if(fceu->FSettings.soundq>=1)
  {
   int32 *tmpo=&WaveHi[soundtsoffs];

   if(GameExpSound.HiFill) (*GameExpSound.HiFill)();

   for(x=fceu->x6502.soundtimestamp;x;x--)
   {
    uint32 b=*tmpo;
    *tmpo=(b&65535)+wlookup2[(b>>16)&255]+wlookup1[b>>24];
    tmpo++;
   }
   end=NeoFilterSound(WaveHi,WaveFinal,SOUNDTS,&left);

   memmove(WaveHi,WaveHi+SOUNDTS-left,left*sizeof(uint32));
   memset(WaveHi+left,0,sizeof(WaveHi)-left*sizeof(uint32));

   if(GameExpSound.HiSync) (*GameExpSound.HiSync)(left);
   for(x=0;x<5;x++)
    ChannelBC[x]=left;
  }
  else
  {
   end=(SOUNDTS<<16)/soundtsinc;
   if(GameExpSound.Fill)
    (*GameExpSound.Fill)(end&0xF);

   SexyFilter(Wave,WaveFinal,end>>4);

   //if(fceu->FSettings.lowpass)
   // SexyFilter2(WaveFinal,end>>4);
   if(end&0xF)
    Wave[0]=Wave[(end>>4)];
   Wave[end>>4]=0;
  }
  nosoundo:

  if(fceu->FSettings.soundq>=1)
  {
   soundtsoffs=left;
  }
  else
  {
   for(x=0;x<5;x++)
    ChannelBC[x]=end&0xF;
   soundtsoffs = (soundtsinc*(end&0xF))>>16;
   end>>=4;
  }
  inbuf=end;

  WriteWaveData(WaveFinal, end); /* This function will just return
				    if sound recording is off. */
  return(end);
}

int Sound::GetSoundBuffer(int32 **W)
{
 *W=WaveFinal;
 return(inbuf);
}

/* FIXME:  Find out what sound registers get reset on reset.  I know $4001/$4005 don't,
due to that whole MegaMan 2 Game Genie thing.
*/

void Sound::Reset(void)
{
	int x;

	IRQFrameMode=0x0;
	fhcnt=fhinc;
	fcnt=0;
	nreg=1;

	for(x=0;x<2;x++)
	{
		wlcount[x]=2048;
		if(nesincsize) // lq mode
			sqacc[x]=((uint32)2048<<17)/nesincsize;
		else
			sqacc[x]=1;
		sweepon[x]=0;
		curfreq[x]=0;
	}

	wlcount[2]=1;  //2048;
	wlcount[3]=2048;

	DMCHaveDMA=DMCHaveSample=0;
	SIRQStat=0x00;

	RawDALatch=0x00;
	TriCount=0;
	TriMode=0;
	tristep=0;
	EnabledChannels=0;
	for(x=0;x<4;x++)
	 lengthcount[x]=0;

	DMCAddressLatch=0;
	DMCSizeLatch=0;
	DMCFormat=0;
	DMCAddress=0;
	DMCSize=0;
	DMCShift=0;

	// MAJOR BUG WAS HERE: DMCacc and DMCBitCount never got reset...
	// so, do some ridiculous hackery if a movie's about to play to keep it in sync...


	if(movieSyncHackOn)
	{
		if(resetDMCacc)
		{
			// no value in movie save state
		#ifdef WIN32
			// use editbox fields
			DMCacc=movieConvertOffset1;
			DMCBitCount=movieConvertOffset2;
		#else
			// no editbox fields, so leave the values alone
			// and print out a warning that says what they are
            fceu::PrintError("Warning: These variables were not found in the save state and will keep their current value: DMCacc=%d, DMCBitCount=%d\n", DMCacc, DMCBitCount);
		#endif
		}
		else
		{
			// keep values loaded from movie save state or reset earlier
		}
	}
	else
	{
		// reset these variables like should have done in the first place
		DMCacc=1;
		DMCBitCount=0;
	}

//	fceu::PrintError("DMCacc=%d, DMCBitCount=%d",DMCacc,DMCBitCount);
}

void Sound::Power(void)
{
        int x;

        SetNESSoundMap();
        memset(PSG,0x00,sizeof(PSG));
	Reset();

	memset(Wave,0,sizeof(Wave));
        memset(WaveHi,0,sizeof(WaveHi));
	memset(&EnvUnits,0,sizeof(EnvUnits));

        for(x=0;x<5;x++)
         ChannelBC[x]=0;
        soundtsoffs=0;
        LoadDMCPeriod(DMCFormat&0xF);
}


void Sound::SetSoundVariables(void)
{
  int x;

  fhinc=fceu->PAL?16626:14915;  // *2 CPU clock rate
  fhinc*=24;

  if(fceu->FSettings.SndRate)
  {
   wlookup1[0]=0;
   for(x=1;x<32;x++)
   {
    wlookup1[x]=(double)16*16*16*4*95.52/((double)8128/(double)x+100);
    if(!fceu->FSettings.soundq) wlookup1[x]>>=4;
   }
   wlookup2[0]=0;
   for(x=1;x<203;x++)
   {
    wlookup2[x]=(double)16*16*16*4*163.67/((double)24329/(double)x+100);
    if(!fceu->FSettings.soundq) wlookup2[x]>>=4;
   }
   if(fceu->FSettings.soundq>=1)
   {
    DoNoise=&RDoNoise_;
    DoTriangle=&RDoTriangle_;
    DoPCM=&RDoPCM_;
    DoSQ1=&RDoSQ1_;
    DoSQ2=&RDoSQ2_;
   }
   else
   {
    DoNoise=DoTriangle=DoPCM=DoSQ1=DoSQ2=&Dummyfunc;
    DoSQ1=&RDoSQLQ_;
    DoSQ2=&RDoSQLQ_;
    DoTriangle=&RDoTriangleNoisePCMLQ_;
    DoNoise=&RDoTriangleNoisePCMLQ_;
    DoPCM=&RDoTriangleNoisePCMLQ_;
   }
  }
  else
  {
   DoNoise=DoTriangle=DoPCM=DoSQ1=DoSQ2=&Dummyfunc;
   return;
  }

  MakeFilters(fceu->FSettings.SndRate);

  if(GameExpSound.RChange)
   (*GameExpSound.RChange)();

  nesincsize=(int64)(((int64)1<<17)*(double)(fceu->PAL?PAL_CPU:NTSC_CPU)/(fceu->FSettings.SndRate * 16));
  memset(sqacc,0,sizeof(sqacc));
  memset(ChannelBC,0,sizeof(ChannelBC));

  LoadDMCPeriod(DMCFormat&0xF);  // For changing from PAL to NTSC

  soundtsinc=(uint32)((uint64)(fceu->PAL?(long double)PAL_CPU*65536:(long double)NTSC_CPU*65536)/(fceu->FSettings.SndRate * 16));
}

void Sound::SetRate(int Rate)
{
	fceu->FSettings.SndRate=Rate;
	SetSoundVariables();
}

void Sound::SetLowPass(int q)
{
	fceu->FSettings.lowpass=q;
}

void Sound::SetSoundQuality(int quality)
{
	fceu->FSettings.soundq=quality;
	SetSoundVariables();
}

void Sound::SetSoundVolume(uint32 volume)
{
	fceu->FSettings.SoundVolume=volume;
}

void Sound::SetTriangleVolume(uint32 volume)
{
	fceu->FSettings.TriangleVolume=volume;
}

void Sound::SetSquare1Volume(uint32 volume)
{
	fceu->FSettings.Square1Volume=volume;
}

void Sound::SetSquare2Volume(uint32 volume)
{
	fceu->FSettings.Square2Volume=volume;
}

void Sound::SetNoiseVolume(uint32 volume)
{
	fceu->FSettings.NoiseVolume=volume;
}

void Sound::SetPCMVolume(uint32 volume)
{
	fceu->FSettings.PCMVolume=volume;
}

void Sound::SaveState(void)
{

}

void Sound::LoadState(int version)
{
 LoadDMCPeriod(DMCFormat&0xF);
 RawDALatch&=0x7F;
 DMCAddress&=0x7FFF;
}

} // namespace fceu
