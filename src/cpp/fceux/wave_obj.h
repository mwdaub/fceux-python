#ifndef _WAVE_H_
#define _WAVE_H_

#include "types_obj.h"

namespace fceu {

void WriteWaveData(int32 *Buffer, int Count);
int EndWaveRecord();

} // namespace fceu

#endif // define _WAVE_H_
