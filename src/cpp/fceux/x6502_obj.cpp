/* FCE Ultra - NES/Famicom Emulator
 *
 * Copyright notice for this file:
 *  Copyright (C) 2002 Xodnizel
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

#include "x6502_obj.h"

namespace FCEU {

static uint8 CycTable[256] = {
/*0x00*/ 7,6,2,8,3,3,5,5,3,2,2,2,4,4,6,6,
/*0x10*/ 2,5,2,8,4,4,6,6,2,4,2,7,4,4,7,7,
/*0x20*/ 6,6,2,8,3,3,5,5,4,2,2,2,4,4,6,6,
/*0x30*/ 2,5,2,8,4,4,6,6,2,4,2,7,4,4,7,7,
/*0x40*/ 6,6,2,8,3,3,5,5,3,2,2,2,3,4,6,6,
/*0x50*/ 2,5,2,8,4,4,6,6,2,4,2,7,4,4,7,7,
/*0x60*/ 6,6,2,8,3,3,5,5,4,2,2,2,5,4,6,6,
/*0x70*/ 2,5,2,8,4,4,6,6,2,4,2,7,4,4,7,7,
/*0x80*/ 2,6,2,6,3,3,3,3,2,2,2,2,4,4,4,4,
/*0x90*/ 2,6,2,6,4,4,4,4,2,5,2,5,5,5,5,5,
/*0xA0*/ 2,6,2,6,3,3,3,3,2,2,2,2,4,4,4,4,
/*0xB0*/ 2,5,2,5,4,4,4,4,2,4,2,4,4,4,4,4,
/*0xC0*/ 2,6,2,8,3,3,5,5,2,2,2,2,4,4,6,6,
/*0xD0*/ 2,5,2,8,4,4,6,6,2,4,2,7,4,4,7,7,
/*0xE0*/ 2,6,3,8,3,3,5,5,2,2,2,2,4,4,6,6,
/*0xF0*/ 2,5,2,8,4,4,6,6,2,4,2,7,4,4,7,7,
};

uint8 X6502::DMR(uint32 A) {
  ADDCYC(1);
  return (DB_=(*ARead)[A](ppu, A));
}

void X6502::DMW(uint32 A, uint8 V) {
  ADDCYC(1);
  (*BWrite)[A](ppu,A,V);
}

/* Some of these operations will only make sense if you know what the flag
   constants are. */

#define X_ZN(zort)      P_&=~(Z_FLAG|N_FLAG);P_|=ZNTable[zort]
#define X_ZNT(zort)  P_|=ZNTable[zort]

#define JR(cond);  \
{    \
 if(cond)  \
 {  \
  uint32 tmp;  \
  int32 disp;  \
  disp=(int8)RdMem(PC_);  \
  PC_++;  \
  ADDCYC(1);  \
  tmp=PC_;  \
  PC_+=disp;  \
  if((tmp^PC_)&0x100)  \
  ADDCYC(1);  \
 }  \
 else PC_++;  \
}


#define LDA     A_=x;X_ZN(A_)
#define LDX     X_=x;X_ZN(X_)
#define LDY  Y_=x;X_ZN(Y_)

/*  All of the freaky arithmetic operations. */
#define AND  A_&=x;X_ZN(A_)
#define BIT  P_&=~(Z_FLAG|V_FLAG|N_FLAG);P_|=ZNTable[x&A_]&Z_FLAG;P_|=x&(V_FLAG|N_FLAG)
#define EOR  A_^=x;X_ZN(A_)
#define ORA  A_|=x;X_ZN(A_)

#define ADC  {  \
        uint32 l=A_+x+(P_&1);  \
        P_&=~(Z_FLAG|C_FLAG|N_FLAG|V_FLAG);  \
        P_|=((((A_^x)&0x80)^0x80) & ((A_^l)&0x80))>>1;  \
        P_|=(l>>8)&C_FLAG;  \
        A_=l;  \
        X_ZNT(A_);  \
       }

#define SBC  {  \
        uint32 l=A_-x-((P_&1)^1);  \
        P_&=~(Z_FLAG|C_FLAG|N_FLAG|V_FLAG);  \
        P_|=((A_^l)&(A_^x)&0x80)>>1;  \
        P_|=((l>>8)&C_FLAG)^C_FLAG;  \
        A_=l;  \
        X_ZNT(A_);  \
       }

#define CMPL(a1,a2) {  \
         uint32 t=a1-a2;  \
         X_ZN(t&0xFF);  \
         P_&=~C_FLAG;  \
         P_|=((t>>8)&C_FLAG)^C_FLAG;  \
		    }

/* Special undocumented operation.  Very similar to CMP. */
#define AXS      {  \
                     uint32 t=(A_&X_)-x;    \
                     X_ZN(t&0xFF);      \
                     P_&=~C_FLAG;       \
         P_|=((t>>8)&C_FLAG)^C_FLAG;  \
         X_=t;  \
        }

#define CMP    CMPL(A_,x)
#define CPX    CMPL(X_,x)
#define CPY          CMPL(Y_,x)

/* The following operations modify the byte being worked on. */
#define DEC         x--;X_ZN(x)
#define INC    x++;X_ZN(x)

#define ASL  P_&=~C_FLAG;P_|=x>>7;x<<=1;X_ZN(x)
#define LSR  P_&=~(C_FLAG|N_FLAG|Z_FLAG);P_|=x&1;x>>=1;X_ZNT(x)

/* For undocumented instructions, maybe for other things later... */
#define LSRA  P_&=~(C_FLAG|N_FLAG|Z_FLAG);P_|=A_&1;A_>>=1;X_ZNT(A_)

#define ROL  {  \
     uint8 l=x>>7;  \
     x<<=1;  \
     x|=P_&C_FLAG;  \
     P_&=~(Z_FLAG|N_FLAG|C_FLAG);  \
     P_|=l;  \
     X_ZNT(x);  \
    }
#define ROR  {  \
     uint8 l=x&1;  \
     x>>=1;  \
     x|=(P_&C_FLAG)<<7;  \
     P_&=~(Z_FLAG|N_FLAG|C_FLAG);  \
     P_|=l;  \
     X_ZNT(x);  \
		}

/* Icky icky thing for some undocumented instructions.  Can easily be
   broken if names of local variables are changed.
*/

/* Absolute */
#define GetAB(target)   \
{  \
 target=RdMem(PC_);  \
 PC_++;  \
 target|=RdMem(PC_)<<8;  \
 PC_++;  \
}

/* Absolute Indexed(for reads) */
#define GetABIRD(target, i)  \
{  \
 unsigned int tmp;  \
 GetAB(tmp);  \
 target=tmp;  \
 target+=i;  \
 if((target^tmp)&0x100)  \
 {  \
  target&=0xFFFF;  \
  RdMem(target^0x100);  \
  ADDCYC(1);  \
 }  \
}

/* Absolute Indexed(for writes and rmws) */
#define GetABIWR(target, i)  \
{  \
 unsigned int rt;  \
 GetAB(rt);  \
 target=rt;  \
 target+=i;  \
 target&=0xFFFF;  \
 RdMem((target&0x00FF)|(rt&0xFF00));  \
}

/* Zero Page */
#define GetZP(target)  \
{  \
 target=RdMem(PC_);   \
 PC_++;  \
}

/* Zero Page Indexed */
#define GetZPI(target,i)  \
{  \
 target=i+RdMem(PC_);  \
 PC_++;  \
}

/* Indexed Indirect */
#define GetIX(target)  \
{  \
 uint8 tmp;  \
 tmp=RdMem(PC_);  \
 PC_++;  \
 tmp+=X_;  \
 target=RdRAM(tmp);  \
 tmp++;    \
 target|=RdRAM(tmp)<<8;  \
}

/* Indirect Indexed(for reads) */
#define GetIYRD(target)  \
{  \
 unsigned int rt;  \
 uint8 tmp;  \
 tmp=RdMem(PC_);  \
 PC_++;  \
 rt=RdRAM(tmp);  \
 tmp++;  \
 rt|=RdRAM(tmp)<<8;  \
 target=rt;  \
 target+=Y_;  \
 if((target^rt)&0x100)  \
 {  \
  target&=0xFFFF;  \
  RdMem(target^0x100);  \
  ADDCYC(1);  \
 }  \
}

/* Indirect Indexed(for writes and rmws) */
#define GetIYWR(target)  \
{  \
 unsigned int rt;  \
 uint8 tmp;  \
 tmp=RdMem(PC_);  \
 PC_++;  \
 rt=RdRAM(tmp);  \
 tmp++;  \
 rt|=RdRAM(tmp)<<8;  \
 target=rt;  \
 target+=Y_;  \
 target&=0xFFFF; \
 RdMem((target&0x00FF)|(rt&0xFF00));  \
}

/* Now come the macros to wrap up all of the above stuff addressing mode functions
   and operation macros.  Note that operation macros will always operate(redundant
   redundant) on the variable "x".
*/

#define RMW_A(op) {uint8 x=A_; op; A_=x; break; } /* Meh... */
#define RMW_AB(op) {unsigned int A; uint8 x; GetAB(A); x=RdMem(A); WrMem(A,x); op; WrMem(A,x); break; }
#define RMW_ABI(reg,op) {unsigned int A; uint8 x; GetABIWR(A,reg); x=RdMem(A); WrMem(A,x); op; WrMem(A,x); break; }
#define RMW_ABX(op)  RMW_ABI(X_,op)
#define RMW_ABY(op)  RMW_ABI(Y_,op)
#define RMW_IX(op)  {unsigned int A; uint8 x; GetIX(A); x=RdMem(A); WrMem(A,x); op; WrMem(A,x); break; }
#define RMW_IY(op)  {unsigned int A; uint8 x; GetIYWR(A); x=RdMem(A); WrMem(A,x); op; WrMem(A,x); break; }
#define RMW_ZP(op)  {uint8 A; uint8 x; GetZP(A); x=RdRAM(A); op; WrRAM(A,x); break; }
#define RMW_ZPX(op) {uint8 A; uint8 x; GetZPI(A,X_); x=RdRAM(A); op; WrRAM(A,x); break;}

#define LD_IM(op)  {uint8 x; x=RdMem(PC_); PC_++; op; break;}
#define LD_ZP(op)  {uint8 A; uint8 x; GetZP(A); x=RdRAM(A); op; break;}
#define LD_ZPX(op)  {uint8 A; uint8 x; GetZPI(A,X_); x=RdRAM(A); op; break;}
#define LD_ZPY(op)  {uint8 A; uint8 x; GetZPI(A,Y_); x=RdRAM(A); op; break;}
#define LD_AB(op)  {unsigned int A; uint8 x; GetAB(A); x=RdMem(A); op; break; }
#define LD_ABI(reg,op)  {unsigned int A; uint8 x; GetABIRD(A,reg); x=RdMem(A); op; break;}
#define LD_ABX(op)  LD_ABI(X_,op)
#define LD_ABY(op)  LD_ABI(Y_,op)
#define LD_IX(op)  {unsigned int A; uint8 x; GetIX(A); x=RdMem(A); op; break;}
#define LD_IY(op)  {unsigned int A; uint8 x; GetIYRD(A); x=RdMem(A); op; break;}

#define ST_ZP(r)  {uint8 A; GetZP(A); WrRAM(A,r); break;}
#define ST_ZPX(r)  {uint8 A; GetZPI(A,X_); WrRAM(A,r); break;}
#define ST_ZPY(r)  {uint8 A; GetZPI(A,Y_); WrRAM(A,r); break;}
#define ST_AB(r)  {unsigned int A; GetAB(A); WrMem(A,r); break;}
#define ST_ABI(reg,r)  {unsigned int A; GetABIWR(A,reg); WrMem(A,r); break; }
#define ST_ABX(r)  ST_ABI(X_,r)
#define ST_ABY(r)  ST_ABI(Y_,r)
#define ST_IX(r)  {unsigned int A; GetIX(A); WrMem(A,r); break; }
#define ST_IY(r)  {unsigned int A; GetIYWR(A); WrMem(A,r); break; }

void X6502::IRQBegin(int w) {
  IRQlow|=w;
}

void X6502::IRQEnd(int w) {
  IRQlow&=~w;
}

void X6502::TriggerNMI(void) {
  IRQlow|=FCEU_IQNMI;
}

void X6502::TriggerNMI2(void) {
  IRQlow|=FCEU_IQNMI2;
}

void X6502::Reset(void) {
  IRQlow=FCEU_IQRESET;
}
/**
* Initializes the 6502 CPU
**/
void X6502::Init(void) {
  unsigned int i;

  for(i = 0; i < sizeof(ZNTable); i++) {
    if(!i) {
      ZNTable[i] = Z_FLAG;
    } else if ( i & 0x80 ) {
      ZNTable[i] = N_FLAG;
    } else {
      ZNTable[i] = 0;
    }
  }
}

void X6502::Power(void)
{
  count_ = tcount = IRQlow = PC_ = A_ = X_ = Y_ = P_ = mooPI = DB_ = jammed = 0;
  S_=0xFD;
  timestamp_=soundtimestamp=0;
  Reset();
  StackAddrBackup = -1;
}

void X6502::Run(int32 cycles) {
  if(PAL)
    cycles*=15;    // 15*4=60
  else
    cycles*=16;    // 16*4=64

  count_ += cycles;
  while(count_ > 0) {
    int32 temp;
    uint8 b1;

    if(IRQlow) {
      if(IRQlow&FCEU_IQRESET) {
	    DEBUG( if(debug_loggingCD) LogCDVectors(0xFFFC); )
        PC_=RdMem(0xFFFC);
        PC_|=RdMem(0xFFFD)<<8;
        jammed=0;
        mooPI = P_ = I_FLAG;
        IRQlow &= ~FCEU_IQRESET;
      } else if(IRQlow&FCEU_IQNMI2) {
        IRQlow&=~FCEU_IQNMI2;
        IRQlow|=FCEU_IQNMI;
      } else if (IRQlow & FCEU_IQNMI) {
        if(!jammed) {
          ADDCYC(7);
          PUSH(PC_ >> 8);
          PUSH(PC_);
          PUSH((P_ & ~B_FLAG) | (U_FLAG));
          P_|=I_FLAG;
	      DEBUG( if(debug_loggingCD) LogCDVectors(0xFFFA) );
          PC_=RdMem(0xFFFA);
          PC_|=RdMem(0xFFFB)<<8;
          IRQlow&=~FCEU_IQNMI;
        }
      } else {
        if(!(mooPI & I_FLAG) && !jammed) {
          ADDCYC(7);
          PUSH(PC_ >> 8);
          PUSH(PC_);
          PUSH((P_ & ~B_FLAG) | (U_FLAG));
          P_ |= I_FLAG;
	      DEBUG( if(debug_loggingCD) LogCDVectors(0xFFFE) );
          PC_ = RdMem(0xFFFE);
          PC_ |= RdMem(0xFFFF)<<8;
        }
      }
      IRQlow &= ~(FCEU_IQTEMP);
      if(count_<=0) {
        mooPI=P_;
        return;
      } //Should increase accuracy without a
              //major speed hit.
    }

	//will probably cause a major speed decrease on low-end systems
    DEBUG( DebugCycle() );

    mooPI=P_;
    b1=RdMem(PC_);

    ADDCYC(CycTable[b1]);

    temp=tcount;
    tcount=0;
    if(MapIRQHook) MapIRQHook(temp);
   
    if (!overclocking)
      FCEU_SoundCPUHook(temp);
    PC_++;
    switch(b1) {
    #include "ops_obj.inc"
    }
  }
}

//the opsize table is used to quickly grab the instruction sizes (in bytes)
const uint8 opsize[256] = {
#ifdef BRK_3BYTE_HACK
/*0x00*/	3, //BRK
#else
/*0x00*/	1, //BRK
#endif
/*0x01*/      2,0,0,0,2,2,0,1,2,1,0,0,3,3,0,
/*0x10*/	2,2,0,0,0,2,2,0,1,3,0,0,0,3,3,0,
/*0x20*/	3,2,0,0,2,2,2,0,1,2,1,0,3,3,3,0,
/*0x30*/	2,2,0,0,0,2,2,0,1,3,0,0,0,3,3,0,
/*0x40*/	1,2,0,0,0,2,2,0,1,2,1,0,3,3,3,0,
/*0x50*/	2,2,0,0,0,2,2,0,1,3,0,0,0,3,3,0,
/*0x60*/	1,2,0,0,0,2,2,0,1,2,1,0,3,3,3,0,
/*0x70*/	2,2,0,0,0,2,2,0,1,3,0,0,0,3,3,0,
/*0x80*/	0,2,0,0,2,2,2,0,1,0,1,0,3,3,3,0,
/*0x90*/	2,2,0,0,2,2,2,0,1,3,1,0,0,3,0,0,
/*0xA0*/	2,2,2,0,2,2,2,0,1,2,1,0,3,3,3,0,
/*0xB0*/	2,2,0,0,2,2,2,0,1,3,1,0,3,3,3,0,
/*0xC0*/	2,2,0,0,2,2,2,0,1,2,1,0,3,3,3,0,
/*0xD0*/	2,2,0,0,0,2,2,0,1,3,0,0,0,3,3,0,
/*0xE0*/	2,2,0,0,2,2,2,0,1,2,1,0,3,3,3,0,
/*0xF0*/	2,2,0,0,0,2,2,0,1,3,0,0,0,3,3,0
};


//the optype table is a quick way to grab the addressing mode for any 6502 opcode
//
//  0 = Implied\Accumulator\Immediate\Branch\NULL
//  1 = (Indirect,X)
//  2 = Zero Page
//  3 = Absolute
//  4 = (Indirect),Y
//  5 = Zero Page,X
//  6 = Absolute,Y
//  7 = Absolute,X
//  8 = Zero Page,Y
//
const uint8 optype[256] = {
/*0x00*/	0,1,0,1,2,2,2,2,0,0,0,0,3,3,3,3,
/*0x10*/	0,4,0,3,5,5,5,5,0,6,0,6,7,7,7,7,
/*0x20*/	0,1,0,1,2,2,2,2,0,0,0,0,3,3,3,3,
/*0x30*/	0,4,0,3,5,5,5,5,0,6,0,6,7,7,7,7,
/*0x40*/	0,1,0,1,2,2,2,2,0,0,0,0,3,3,3,3,
/*0x50*/	0,4,0,3,5,5,5,5,0,6,0,6,7,7,7,7,
/*0x60*/	0,1,0,1,2,2,2,2,0,0,0,0,3,3,3,3,
/*0x70*/	0,4,0,3,5,5,5,5,0,6,0,6,7,7,7,7,
/*0x80*/	0,1,0,1,2,2,2,2,0,0,0,0,3,3,3,3,
/*0x90*/	0,4,0,3,5,5,8,8,0,6,0,6,7,7,6,6,
/*0xA0*/	0,1,0,1,2,2,2,2,0,0,0,0,3,3,3,3,
/*0xB0*/	0,4,0,3,5,5,8,8,0,6,0,6,7,7,6,6,
/*0xC0*/	0,1,0,1,2,2,2,2,0,0,0,0,3,3,3,3,
/*0xD0*/	0,4,0,3,5,5,5,5,0,6,0,6,7,7,7,7,
/*0xE0*/	0,1,0,1,2,2,2,2,0,0,0,0,3,3,3,3,
/*0xF0*/	0,4,0,3,5,5,5,5,0,6,0,6,7,7,7,7,
};

// the opwrite table aids in predicting the value written for any 6502 opcode
//
//  0 = No value written
//  1 = Write from A
//  2 = Write from X
//  3 = Write from Y
//  4 = Write from P
//  5 = ASL (SLO)
//  6 = LSR (SRE)
//  7 = ROL (RLA)
//  8 = ROR (RRA)
//  9 = INC (ISC)
// 10 = DEC (DCP)
// 11 = (SAX)
// 12 = (AHX)
// 13 = (SHY)
// 14 = (SHX)
// 15 = (TAS)

const uint8 opwrite[256] = {
/*0x00*/	 0, 0, 0, 5, 0, 0, 5, 5, 4, 0, 0, 0, 0, 0, 5, 5,
/*0x10*/	 0, 0, 0, 5, 0, 0, 5, 5, 0, 0, 0, 5, 0, 0, 5, 5,
/*0x20*/	 0, 0, 0, 7, 0, 0, 7, 7, 0, 0, 7, 0, 0, 0, 7, 7,
/*0x30*/	 0, 0, 0, 7, 0, 0, 7, 7, 0, 0, 0, 7, 0, 0, 7, 7,
/*0x40*/	 0, 0, 0, 6, 0, 0, 6, 6, 1, 0, 6, 0, 0, 0, 6, 6,
/*0x50*/	 0, 0, 0, 6, 0, 0, 6, 6, 0, 0, 0, 6, 0, 0, 6, 6,
/*0x60*/	 0, 0, 0, 8, 0, 0, 8, 8, 0, 0, 8, 0, 0, 0, 8, 8,
/*0x70*/	 0, 0, 0, 8, 0, 0, 8, 8, 0, 0, 0, 8, 0, 0, 8, 8,
/*0x80*/	 0, 1, 0,11, 3, 1, 2,11, 0, 0, 0, 0, 3, 1, 2,11,
/*0x90*/	 0, 1, 0,12, 3, 1, 2,11, 0, 1, 0,15,13, 1,14,12,
/*0xA0*/	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
/*0xB0*/	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
/*0xC0*/	 0, 0, 0,10, 0, 0,10,10, 0, 0, 0, 0, 0, 0,10,10,
/*0xD0*/	 0, 0, 0,10, 0, 0,10,10, 0, 0, 0,10, 0, 0,10,10,
/*0xE0*/	 0, 0, 0, 9, 0, 0, 9, 9, 0, 0, 0, 0, 0, 0, 9, 9,
/*0xF0*/	 0, 0, 0, 9, 0, 0, 9, 9, 0, 0, 0, 9, 0, 0, 9, 9,
};

}
