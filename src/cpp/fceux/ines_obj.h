#ifndef _INES_H_
#define _INES_H_

#include <stdlib.h>
#include <string.h>
#include <map>

#include "types_obj.h"
#include "git_obj.h"

#include "cart_obj.h"
#include "file_obj.h"

namespace fceu {

struct TMasterRomInfo
{
	uint64 md5lower;
	const char* params;
};

class TMasterRomInfoParams : public std::map<std::string,std::string>
{
public:
	bool ContainsKey(const std::string& key) { return find(key) != end(); }
};

//mbg merge 7/19/06 changed to c++ decl format
struct iNES_HEADER {
	char ID[4]; /*NES^Z*/
	uint8 ROM_size;
	uint8 VROM_size;
	uint8 ROM_type;
	uint8 ROM_type2;
	uint8 ROM_type3;
	uint8 Upper_ROM_VROM_size;
	uint8 RAM_size;
	uint8 VRAM_size;
	uint8 TV_system;
	uint8 VS_hardware;
	uint8 reserved[2];

	void cleanup()
	{
		if(!memcmp((char *)(this)+0x7,"DiskDude",8))
		{
			memset((char *)(this)+0x7,0,0x9);
		}

		if(!memcmp((char *)(this)+0x7,"demiforce",9))
		{
			memset((char *)(this)+0x7,0,0x9);
		}

		if(!memcmp((char *)(this)+0xA,"Ni03",4))
		{
			if(!memcmp((char *)(this)+0x7,"Dis",3))
				memset((char *)(this)+0x7,0,0x9);
			else
				memset((char *)(this)+0xA,0,0x6);
		}
	}
};

class FCEU;

class iNES {
  public:
    //mbg merge 6/29/06 - these need to be global
    uint8 *trainerpoo = NULL;
    uint8 *ROM = NULL;
    uint8 *VROM = NULL;
    uint8 *ExtraNTARAM = NULL;
    iNES_HEADER head;

    CartInfo iNESCart;

    uint8 Mirroring = 0;
    uint32 ROM_size = 0;
    uint32 VROM_size = 0;

    int CHRRAMSize = -1;

    int MapperNo = 0;

    int iNES2 = 0;

    uint32 iNESGameCRC32 = 0;

    const TMasterRomInfo* MasterRomInfo;
    TMasterRomInfoParams MasterRomInfoParams;

    //mbg merge 6/29/06
    int iNesSave(); //bbit Edited: line added
    int iNesSaveAs(char* name);
    int iNESLoad(const char *name, FCEUFILE *fp, int OverwriteVidMode);

    uint8 TrainerRead(uint32 A);

    readfunc TrainerRead_ = [this](uint32 A) { return TrainerRead(A); };

    int iNES_Init(int num);
    void iNES_ExecPower();
    void iNESGI(GI h);
    void SetInput(void);

    std::function<void(GI)> iNESGI_ = [this](GI h) { iNESGI(h); };

    void CheckBad(uint64 md5partial);
    void CheckHInfo(void);

    char* iNesShortFName();

  private:
    FCEU* fceu;
};

void NSFVRC6_Init(void);
void NSFMMC5_Init(void);
void NSFAY_Init(void);
void NSFN106_Init(void);
void NSFVRC7_Init(void);

void Mapper1_Init(CartInfo *);
void Mapper4_Init(CartInfo *);
void Mapper5_Init(CartInfo *);
void Mapper6_Init(CartInfo *);
void Mapper8_Init(CartInfo *);
void Mapper9_Init(CartInfo *);
void Mapper10_Init(CartInfo *);
void Mapper11_Init(CartInfo *);
void Mapper12_Init(CartInfo *);
void Mapper15_Init(CartInfo *);
void Mapper16_Init(CartInfo *);
void Mapper17_Init(CartInfo *);
void Mapper18_Init(CartInfo *);
void Mapper19_Init(CartInfo *);
void Mapper21_Init(CartInfo *);
void Mapper22_Init(CartInfo *);
void Mapper23_Init(CartInfo *);
void Mapper24_Init(CartInfo *);
void Mapper25_Init(CartInfo *);
void Mapper26_Init(CartInfo *);
void Mapper28_Init(CartInfo *);
void Mapper29_Init(CartInfo *);
void Mapper31_Init(CartInfo *);
void Mapper32_Init(CartInfo *);
void Mapper33_Init(CartInfo *);
void Mapper34_Init(CartInfo *);
void Mapper36_Init(CartInfo *);
void Mapper37_Init(CartInfo *);
void Mapper38_Init(CartInfo *);
void Mapper40_Init(CartInfo *);
void Mapper41_Init(CartInfo *);
void Mapper42_Init(CartInfo *);
void Mapper43_Init(CartInfo *);
void Mapper44_Init(CartInfo *);
void Mapper45_Init(CartInfo *);
void Mapper46_Init(CartInfo *);
void Mapper47_Init(CartInfo *);
void Mapper48_Init(CartInfo *);
void Mapper49_Init(CartInfo *);
void Mapper50_Init(CartInfo *);
void Mapper51_Init(CartInfo *);
void Mapper52_Init(CartInfo *);
void Mapper57_Init(CartInfo *);
void Mapper59_Init(CartInfo *);
void Mapper61_Init(CartInfo *);
void Mapper62_Init(CartInfo *);
void Mapper64_Init(CartInfo *);
void Mapper65_Init(CartInfo *);
void Mapper67_Init(CartInfo *);
void Mapper68_Init(CartInfo *);
void Mapper69_Init(CartInfo *);
void Mapper70_Init(CartInfo *);
void Mapper71_Init(CartInfo *);
void Mapper72_Init(CartInfo *);
void Mapper73_Init(CartInfo *);
void Mapper74_Init(CartInfo *);
void Mapper75_Init(CartInfo *);
void Mapper76_Init(CartInfo *);
void Mapper77_Init(CartInfo *);
void Mapper78_Init(CartInfo *);
void Mapper79_Init(CartInfo *);
void Mapper80_Init(CartInfo *);
void Mapper82_Init(CartInfo *);
void Mapper83_Init(CartInfo *);
void Mapper85_Init(CartInfo *);
void Mapper86_Init(CartInfo *);
void Mapper87_Init(CartInfo *);
void Mapper88_Init(CartInfo *);
void Mapper89_Init(CartInfo *);
void Mapper90_Init(CartInfo *);
void Mapper91_Init(CartInfo *);
void Mapper92_Init(CartInfo *);
void Mapper93_Init(CartInfo *);
void Mapper94_Init(CartInfo *);
void Mapper95_Init(CartInfo *);
void Mapper96_Init(CartInfo *);
void Mapper97_Init(CartInfo *);
void Mapper99_Init(CartInfo *);
void Mapper101_Init(CartInfo *);
void Mapper103_Init(CartInfo *);
void Mapper105_Init(CartInfo *);
void Mapper106_Init(CartInfo *);
void Mapper107_Init(CartInfo *);
void Mapper108_Init(CartInfo *);
void Mapper111_Init(CartInfo *);
void Mapper112_Init(CartInfo *);
void Mapper113_Init(CartInfo *);
void Mapper114_Init(CartInfo *);
void Mapper115_Init(CartInfo *);
void Mapper117_Init(CartInfo *);
void Mapper119_Init(CartInfo *);
void Mapper120_Init(CartInfo *);
void Mapper121_Init(CartInfo *);
void Mapper125_Init(CartInfo *);
void Mapper134_Init(CartInfo *);
void Mapper140_Init(CartInfo *);
void Mapper144_Init(CartInfo *);
void Mapper151_Init(CartInfo *);
void Mapper152_Init(CartInfo *);
void Mapper153_Init(CartInfo *);
void Mapper154_Init(CartInfo *);
void Mapper155_Init(CartInfo *);
void Mapper156_Init(CartInfo *);
void Mapper157_Init(CartInfo *);
void Mapper159_Init(CartInfo *);
void Mapper163_Init(CartInfo *);
void Mapper164_Init(CartInfo *);
void Mapper165_Init(CartInfo *);
void Mapper166_Init(CartInfo *);
void Mapper167_Init(CartInfo *);
void Mapper168_Init(CartInfo *);
void Mapper170_Init(CartInfo *);
void Mapper171_Init(CartInfo *);
void Mapper172_Init(CartInfo *);
void Mapper173_Init(CartInfo *);
void Mapper175_Init(CartInfo *);
void Mapper177_Init(CartInfo *);
void Mapper178_Init(CartInfo *);
void Mapper180_Init(CartInfo *);
void Mapper181_Init(CartInfo *);
void Mapper183_Init(CartInfo *);
void Mapper184_Init(CartInfo *);
void Mapper185_Init(CartInfo *);
void Mapper186_Init(CartInfo *);
void Mapper187_Init(CartInfo *);
void Mapper188_Init(CartInfo *);
void Mapper189_Init(CartInfo *);
void Mapper191_Init(CartInfo *);
void Mapper192_Init(CartInfo *);
void Mapper193_Init(CartInfo *);
void Mapper194_Init(CartInfo *);
void Mapper195_Init(CartInfo *);
void Mapper196_Init(CartInfo *);
void Mapper197_Init(CartInfo *);
void Mapper198_Init(CartInfo *);
void Mapper199_Init(CartInfo *);
void Mapper200_Init(CartInfo *);
void Mapper201_Init(CartInfo *);
void Mapper202_Init(CartInfo *);
void Mapper203_Init(CartInfo *);
void Mapper204_Init(CartInfo *);
void Mapper205_Init(CartInfo *);
void Mapper206_Init(CartInfo *);
void Mapper207_Init(CartInfo *);
void Mapper208_Init(CartInfo *);
void Mapper209_Init(CartInfo *);
void Mapper210_Init(CartInfo *);
void Mapper211_Init(CartInfo *);
void Mapper212_Init(CartInfo *);
void Mapper213_Init(CartInfo *);
void Mapper214_Init(CartInfo *);
void Mapper216_Init(CartInfo *);
void Mapper217_Init(CartInfo *);
void Mapper220_Init(CartInfo *);
void Mapper222_Init(CartInfo *);
void Mapper225_Init(CartInfo *);
void Mapper226_Init(CartInfo *);
void Mapper227_Init(CartInfo *);
void Mapper228_Init(CartInfo *);
void Mapper229_Init(CartInfo *);
void Mapper230_Init(CartInfo *);
void Mapper231_Init(CartInfo *);
void Mapper232_Init(CartInfo *);
void Mapper233_Init(CartInfo *);
void Mapper234_Init(CartInfo *);
void Mapper235_Init(CartInfo *);
void Mapper236_Init(CartInfo *);
void Mapper237_Init(CartInfo *);
void Mapper240_Init(CartInfo *);
void Mapper241_Init(CartInfo *);
void Mapper242_Init(CartInfo *);
void Mapper244_Init(CartInfo *);
void Mapper245_Init(CartInfo *);
void Mapper246_Init(CartInfo *);
void Mapper249_Init(CartInfo *);
void Mapper250_Init(CartInfo *);
void Mapper252_Init(CartInfo *);
void Mapper253_Init(CartInfo *);
void Mapper254_Init(CartInfo *);

} // namespace fceu

#endif
