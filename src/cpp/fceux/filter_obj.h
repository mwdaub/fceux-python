#ifndef _FILTER_H_
#define _FILTER_H_

#include "types_obj.h"

namespace fceu {

int32 NeoFilterSound(int32 *in, int32 *out, uint32 inlen, int32 *leftover);
void MakeFilters(int32 rate);
void SexyFilter(int32 *in, int32 *out, int32 count);

} // namespace fceu

#endif // define _FILTER_H_
