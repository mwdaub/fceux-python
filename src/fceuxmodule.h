#include <Python.h>
#include <numpy/arrayobject.h>

#include "fceux/git.h"
#include "fceux/palette.h"

// Forward declaration of required FCEUX functions.
bool FCEUI_Initialize();
void FCEUI_SetInput(int port, ESI type, void *ptr, int attrib);
FCEUGI *FCEUI_LoadGame(const char *name, int OverwriteVidMode, bool silent = false);
void FCEUI_Emulate(unsigned char **pXBuf, signed int **SoundBuf, signed int *SoundBufSize, int skip);
unsigned char GetMem(unsigned short addr);
