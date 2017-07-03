#include <Python.h>
#include <numpy/arrayobject.h>

#include "fceux/git.h"
#include "fceux/movie.h"
#include "fceux/palette.h"

// Forward declaration of required FCEUX functions.
bool FCEUI_Initialize();
void FCEUI_SetInput(int port, ESI type, void *ptr, int attrib);
FCEUGI *FCEUI_LoadGame(const char *name, int OverwriteVidMode, bool silent = false);
void FCEUI_CloseGame();
void FCEUI_SaveMovie(const char *filename, EMOVIE_FLAG flags, std::wstring author);
void FCEUI_Emulate(uint8 **pXBuf, int32 **SoundBuf, int32 *SoundBufSize, int skip);
uint8 GetMem(uint16 addr);
