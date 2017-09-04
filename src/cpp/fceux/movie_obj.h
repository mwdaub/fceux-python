#ifndef _MOVIE_H_
#define _MOVIE_H_

#include "types_obj.h"
#include "git_obj.h"

#include "utils/md5_obj.h"
#include "utils/valuearray_obj.h"

#include "file_obj.h"

#include <vector>
#include <map>
#include <string>
#include <ostream>
#include <cstdlib>

namespace fceu {

enum EMOVIE_FLAG
{
	MOVIE_FLAG_NONE = 0,

	//an ARCHAIC flag which means the movie was recorded from a soft reset.
	//WHY would you do this?? do not create any new movies with this flag
	MOVIE_FLAG_FROM_RESET = (1<<1),

	MOVIE_FLAG_PAL = (1<<2),

	//movie was recorded from poweron. the alternative is from a savestate (or from reset)
	MOVIE_FLAG_FROM_POWERON = (1<<3),

	// set in newer version, used for old movie compatibility
	//TODO - only use this flag to print a warning that the sync might be bad
	//so that we can get rid of the sync hack code
	MOVIE_FLAG_NOSYNCHACK = (1<<4)
};

typedef struct
{
	int movie_version;					// version of the movie format in the file
	uint32 num_frames;
	uint32 rerecord_count;
	bool poweron, pal, nosynchack, ppuflag;
	bool reset; //mbg 6/21/08 - this flag isnt used anymore.. but maybe one day we can scan it out of the first record in the movie file
	uint32 emu_version_used;				// 9813 = 0.98.13
	MD5DATA md5_of_rom_used;
	std::string name_of_rom_used;

	std::vector<std::wstring> comments;
	std::vector<std::string> subtitles;
} MOVIE_INFO;

void FCEUMOV_AddInputState();
void FCEUMOV_AddCommand(int cmd);
void FCEU_DrawMovies(uint8 *);
void FCEU_DrawLagCounter(uint8 *);

enum EMOVIEMODE
{
	MOVIEMODE_INACTIVE = 1,
	MOVIEMODE_RECORD = 2,
	MOVIEMODE_PLAY = 4,
	MOVIEMODE_TASEDITOR = 8,
	MOVIEMODE_FINISHED = 16
};

enum EMOVIECMD
{
	MOVIECMD_RESET = 1,
	MOVIECMD_POWER = 2,
	MOVIECMD_FDS_INSERT = 4,
	MOVIECMD_FDS_SELECT = 8,
	MOVIECMD_VS_INSERTCOIN = 16
};

EMOVIEMODE FCEUMOV_Mode();
bool FCEUMOV_Mode(EMOVIEMODE modemask);
bool FCEUMOV_Mode(int modemask);
inline bool FCEUMOV_IsPlaying() { return (FCEUMOV_Mode(MOVIEMODE_PLAY|MOVIEMODE_FINISHED)); }
inline bool FCEUMOV_IsRecording() { return FCEUMOV_Mode(MOVIEMODE_RECORD); }
inline bool FCEUMOV_IsFinished() { return FCEUMOV_Mode(MOVIEMODE_FINISHED);}
inline bool FCEUMOV_IsLoaded() { return (FCEUMOV_Mode(MOVIEMODE_PLAY|MOVIEMODE_RECORD|MOVIEMODE_FINISHED)); }

bool FCEUMOV_ShouldPause(void);
int FCEUMOV_GetFrame(void);
int FCEUI_GetLagCount(void);
bool FCEUI_GetLagged(void);
void FCEUI_SetLagFlag(bool value);

int FCEUMOV_WriteState(EMUFILE* os);
bool FCEUMOV_ReadState(EMUFILE* is, uint32 size);
void FCEUMOV_PreLoad();
bool FCEUMOV_PostLoad();
void FCEUMOV_IncrementRerecordCount();

bool FCEUMOV_FromPoweron();

void FCEUMOV_CreateCleanMovie();
void FCEUMOV_ClearCommands();

class MovieData;
class MovieRecord
{

public:
	MovieRecord();
	ValueArray<uint8,4> joysticks;

	struct {
		uint8 x,y,b,bogo;
		uint64 zaphit;
	} zappers[2];

	//misc commands like reset, etc.
	//small now to save space; we might need to support more commands later.
	//the disk format will support up to 64bit if necessary
	uint8 commands;
	bool command_reset() { return (commands & MOVIECMD_RESET) != 0; }
	bool command_power() { return (commands & MOVIECMD_POWER) != 0; }
	bool command_fds_insert() { return (commands & MOVIECMD_FDS_INSERT) != 0; }
	bool command_fds_select() { return (commands & MOVIECMD_FDS_SELECT) != 0; }
	bool command_vs_insertcoin() { return (commands & MOVIECMD_VS_INSERTCOIN) != 0; }

	void toggleBit(int joy, int bit)
	{
		joysticks[joy] ^= mask(bit);
	}

	void setBit(int joy, int bit)
	{
		joysticks[joy] |= mask(bit);
	}

	void clearBit(int joy, int bit)
	{
		joysticks[joy] &= ~mask(bit);
	}

	void setBitValue(int joy, int bit, bool val)
	{
		if(val) setBit(joy,bit);
		else clearBit(joy,bit);
	}

	bool checkBit(int joy, int bit)
	{
		return (joysticks[joy] & mask(bit))!=0;
	}

	bool Compare(MovieRecord& compareRec);
	void Clone(MovieRecord& sourceRec);
	void clear();

	void parse(MovieData* md, EMUFILE* is);
	bool parseBinary(MovieData* md, EMUFILE* is);
	void dump(MovieData* md, EMUFILE* os, int index);
	void dumpBinary(MovieData* md, EMUFILE* os, int index);
	void parseJoy(EMUFILE* is, uint8& joystate);
	void dumpJoy(EMUFILE* os, uint8 joystate);

	static const char mnemonics[8];

private:
	int mask(int bit) { return 1<<bit; }
};

int currFrameCounter;

void MakeBackupMovie(bool dispMessage);
void CreateMovieFile(std::string fn);
void SaveMovie(const char *fname, EMOVIE_FLAG flags, std::wstring author);
bool LoadMovie(const char *fname, bool read_only, int _stopframe);
void MoviePlayFromBeginning(void);
void StopMovie(void);
bool MovieGetInfo(FCEUFILE* fp, MOVIE_INFO& info, bool skipFrameCount = false);
void MovieToggleReadOnly(void);
bool GetMovieToggleReadOnly();
void SetMovieToggleReadOnly(bool which);
int GetMovieLength();
int GetMovieRerecordCount();
std::string GetMovieName(void);
void MovieToggleFrameDisplay();
void MovieToggleRerecordDisplay();
void ToggleInputDisplay(void);

void LoadSubtitles(MovieData &);
void ProcessSubtitles(void);
void DisplaySubtitles(char *format, ...);

void poweron(bool shouldDisableBatteryLoading);

} // namespace fceu

#endif // define _MOVIE_H_
