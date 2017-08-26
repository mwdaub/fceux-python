#ifndef _STATEH
#define _STATEH

//indicates that the value is a multibyte integer that needs to be put in the correct byte order
#define FCEUSTATE_RLSB            0x80000000

//void*v is actually a void** which will be indirected before reading
#define FCEUSTATE_INDIRECT            0x40000000

//all FCEUSTATE flags together so that we can mask them out and get the size
#define FCEUSTATE_FLAGS (FCEUSTATE_RLSB|FCEUSTATE_INDIRECT)

namespace FCEU {

enum ENUM_SSLOADPARAMS
{
	SSLOADPARAM_NOBACKUP,
	SSLOADPARAM_BACKUP,
};

struct SFORMAT
{
	//a void* to the data or a void** to the data
	void *v;

	//size, plus flags
	uint32 s;

	//a string description of the element
	char *desc;
};

 //zlib values: 0 (none) through 9 (max) or -1 (default)
bool FCEUSS_SaveMS(EMUFILE* outstream, int compressionLevel);

bool FCEUSS_LoadFP(EMUFILE* is, ENUM_SSLOADPARAMS params);

void FCEUSS_Save(const char *, bool display_message=true);
bool FCEUSS_Load(const char *, bool display_message=true);

char lastSavestateMade[2048]; //Stores the filename of the last savestate made (needed for UndoSavestate)
bool undoSS = false;		  //This will be true if there is lastSavestateMade, it was made since ROM was loaded, a backup state for lastSavestateMade exists
bool redoSS = false;		  //This will be true if UndoSaveState is run, will turn false when a new savestate is made

char lastLoadstateMade[2048]; //Stores the filename of the last state loaded (needed for Undo/Redo loadstate)
bool undoLS = false;		  //This will be true if a backupstate was made and it was made since ROM was loaded
bool redoLS = false;		  //This will be true if a backupstate was loaded, meaning redoLoadState can be run

bool CheckBackupSaveStateExist();	 //Checks if backupsavestate exists

bool backupSavestates = true;
bool compressSavestates = true;  //By default FCEUX compresses savestates when a movie is inactive.

void ResetExState(void (*PreSave)(void),void (*PostSave)(void));
void AddExState(void *v, uint32 s, int type, char *desc);

void FCEU_DrawSaveStates(uint8 *XBuf);

void CreateBackupSaveState(const char *fname); //backsup a savestate before overwriting it with a new one
void BackupLoadState();				 //Makes a backup savestate before any loadstate
void LoadBackup();					 //Loads the backupsavestate
void RedoLoadState();				 //reloads a loadstate if backupsavestate was run
void SwapSaveState();				 //Swaps a savestate with its backup state

} // namespace FCEU

#endif // define _STATEH
