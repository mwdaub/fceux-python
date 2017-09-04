#ifndef _CHEAT_H_
#define _CHEAT_H_

namespace fceu {

void CheatResetRAM(void);
void CheatAddRAM(int s, uint32 A, uint8 *p);

void LoadGameCheats(FILE *override);
void FlushGameCheats(FILE *override, int nosave);
void ApplyPeriodicCheats(void);
void PowerCheats(void);

int CheatGetByte(uint32 A);
void CheatSetByte(uint32 A, uint8 V);

} // namespace fceu

#endif // define _CHEAT_H_
