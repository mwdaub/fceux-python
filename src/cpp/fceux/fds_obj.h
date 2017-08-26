#ifndef _FDS_H_
#define _FDS_H_

#include "file_obj.h"

namespace FCEU {

int FDSLoad(const char *name, FCEUFILE *fp);

bool isFDS;
void FDSSoundReset(void);

void FCEU_FDSInsert(void);
//void FCEU_FDSEject(void);
void FCEU_FDSSelect(void);

} // namespace FCEU

#endif // define _FDS_H_
