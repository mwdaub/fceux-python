#ifndef _CHEAT_H_
#define _CHEAT_H_

#define CHEATC_NONE     0x8000
#define CHEATC_EXCLUDED 0x4000
#define CHEATC_NOSHOW   0xC000

#include "types_obj.h"

#include "file_obj.h"

namespace fceu {

struct CHEATF {
	struct CHEATF *next;
	char *name;
	uint16 addr;
	uint8 val;
	int compare;	/* -1 for no compare. */
	int type;	/* 0 for replace, 1 for substitute(GG). */
	int status;
};

typedef struct {
	uint16 addr;
	uint8 val;
	int compare;
	readfunc* PrevRead;
} CHEATF_SUBFAST;

class FCEU;

class Cheat {
  public:
    // Methods.
    void CheatResetRAM(void);
    void CheatAddRAM(int s, uint32 A, uint8 *p);

    void LoadGameCheats(FILE *override);
    void FlushGameCheats(FILE *override, int nosave);
    void ApplyPeriodicCheats(void);
    void PowerCheats(void);

    int CheatGetByte(uint32 A);
    void CheatSetByte(uint32 A, uint8 V);

  private:
    // Members.
    FCEU* fceu;

    static uint8 *CheatRPtrs[64];

    std::vector<uint16> FrozenAddresses;			//List of addresses that are currently frozen
    unsigned int FrozenAddressCount=0;		//Keeps up with the Frozen address count, necessary for using in other dialogs (such as hex editor)

    CHEATF_SUBFAST SubCheats[256];
    int numsubcheats=0;
    struct CHEATF *cheats=0,*cheatsl=0;

    uint16 *CheatComp = 0;
    int savecheats = 0;

    readfunc SubCheatsRead_ = [this](uint32 A) { return SubCheatsRead(A); };

    // Methods.
    void UpdateFrozenList(void);			//Function that populates the list of frozen addresses

    uint8 SubCheatsRead(uint32 A);
    void RebuildSubCheats(void);

    void CheatMemErr(void);
    int AddCheatEntry(char *name, uint32 addr, uint8 val, int compare, int status, int type);
    int AddCheat(const char *name, uint32 addr, uint8 val, int compare, int type);
    int DelCheat(uint32 which);
    void ListCheats(int (*callb)(char *name, uint32 a, uint8 v, int compare, int s, int type, void *data), void *data);
    int GetCheat(uint32 which, char **name, uint32 *a, uint8 *v, int *compare, int *s, int *type);
    int SetCheat(uint32 which, const char *name, int32 a, int32 v, int c, int s, int type);
    int ToggleCheat(uint32 which);

    int GGtobin(char c);
    int DecodeGG(const char *str, int *a, int *v, int *c);
    int DecodePAR(const char *str, int *a, int *v, int *c, int *type);

    int InitCheatComp(void);

    void CheatSearchSetCurrentAsOriginal(void);
    void CheatSearchShowExcluded(void);
    int32 CheatSearchGetCount(void);
    void CheatSearchGet(int (*callb)(uint32 a, uint8 last, uint8 current, void *data),void *data);
    void CheatSearchGetRange(uint32 first, uint32 last, int (*callb)(uint32 a, uint8 last, uint8 current));
    void CheatSearchBegin(void);
    void CheatSearchEnd(int type, uint8 v1, uint8 v2);
};

} // namespace fceu

#endif // define _CHEAT_H_
