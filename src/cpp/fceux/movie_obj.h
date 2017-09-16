#ifndef _MOVIE_H_
#define _MOVIE_H_

#include <vector>
#include <map>
#include <string>
#include <ostream>
#include <cstdlib>

#include "types_obj.h"
#include "git_obj.h"
#include "version_obj.h"
#include "video_obj.h"

#include "utils/general_obj.h"
#include "utils/guid_obj.h"
#include "utils/md5_obj.h"
#include "utils/valuearray_obj.h"
#include "utils/xstring_obj.h"

#include "drivers/videolog/nesvideos-piece_obj.h"

#include "cart_obj.h"
#include "drawing_obj.h"
#include "file_obj.h"
#include "state_obj.h"

#define MOVIE_VERSION           3

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

class MovieData
{
public:
	MovieData();
	// Default Values: MovieData::MovieData()

	int version;
	int emuVersion;
	int fds;
	//todo - somehow force mutual exclusion for poweron and reset (with an error in the parser)
	bool palFlag;
	bool PPUflag;
	MD5DATA romChecksum;
	std::string romFilename;
	std::vector<uint8> savestate;
	std::vector<MovieRecord> records;
	std::vector<std::wstring> comments;
	std::vector<std::string> subtitles;
	//this is the RERECORD COUNT. please rename variable.
	int rerecordCount;
	Guid guid;

	//was the frame data stored in binary?
	bool binaryFlag;
	// TAS Editor project files contain additional data after input
	int loadFrameCount;

	//which ports are defined for the movie
	int ports[2];
	//whether fourscore is enabled
	bool fourscore;
	//whether microphone is enabled
	bool microphone;

	int getNumRecords() { return records.size(); }

	class TDictionary : public std::map<std::string,std::string>
	{
	public:
		bool containsKey(std::string key)
		{
			return find(key) != end();
		}

		void tryInstallBool(std::string key, bool& val)
		{
			if(containsKey(key))
				val = atoi(operator [](key).c_str())!=0;
		}

		void tryInstallString(std::string key, std::string& val)
		{
			if(containsKey(key))
				val = operator [](key);
		}

		void tryInstallInt(std::string key, int& val)
		{
			if(containsKey(key))
				val = atoi(operator [](key).c_str());
		}

	};

	void truncateAt(int frame);
	void installValue(std::string& key, std::string& val);
	int dump(EMUFILE* os, bool binary);

	void clearRecordRange(int start, int len);
	void eraseRecords(int at, int frames = 1);
	void insertEmpty(int at, int frames);
	void cloneRegion(int at, int frames);

	static bool loadSavestateFrom(std::vector<uint8>* buf);
	static void dumpSavestateTo(std::vector<uint8>* buf, int compressionLevel);

private:
	void installInt(std::string& val, int& var)
	{
		var = atoi(val.c_str());
	}

	void installBool(std::string& val, bool& var)
	{
		var = atoi(val.c_str())!=0;
	}
};

class FCEU;

class Movie {
  public:

    int WriteState(EMUFILE* os);
    bool ReadState(EMUFILE* is, uint32 size);
    void PreLoad();
    bool PostLoad();
    void IncrementRerecordCount();

    EMOVIEMODE Mode();
    bool Mode(EMOVIEMODE modemask);
    bool Mode(int modemask);
    inline bool IsPlaying() { return (Mode(MOVIEMODE_PLAY|MOVIEMODE_FINISHED)); }
    inline bool IsRecording() { return Mode(MOVIEMODE_RECORD); }
    inline bool IsFinished() { return Mode(MOVIEMODE_FINISHED);}
    inline bool IsLoaded() { return (Mode(MOVIEMODE_PLAY|MOVIEMODE_RECORD|MOVIEMODE_FINISHED)); }

    bool ShouldPause(void);
    int GetFrame(void);
    void SetFrame(int frame) { currFrameCounter = frame; };
    int GetLagCount(void);
    void IncreaseLagCount(void);
    bool GetLagged(void);
    void SetLagFlag(bool value);
    void LagCounterReset();
    void LagCounterToggle(void);
    void FA_SkipLag(void);
    bool GetFrameAdvanceLagSkip(void);
    void SubtitleToggle(void);

    void CreateCleanMovie();
    void ClearCommands();

    bool FromPoweron();

    void AddInputState();
    void AddCommand(int cmd);
    void DrawMovies(uint8 *);
    void DrawLagCounter(uint8 *);

    void MakeBackup(bool dispMessage);
    void CreateFile(std::string fn);
    void Save(const char *fname, EMOVIE_FLAG flags, std::wstring author);
    bool Load(const char *fname, bool read_only, int _stopframe);
    void PlayFromBeginning(void);
    void Stop(void);
    bool GetInfo(FCEUFILE* fp, MOVIE_INFO& info, bool skipFrameCount = false);
    void ToggleReadOnly(void);
    bool GetToggleReadOnly();
    void SetToggleReadOnly(bool which);
    int GetLength();
    int GetRerecordCount();
    std::string GetName(void);
    void ToggleFrameDisplay();
    void ToggleRerecordDisplay();
    void ToggleInputDisplay(void);

    void LoadSubtitles(MovieData &);
    void ProcessSubtitles(void);
    void DisplaySubtitles(char *format, ...);

    void poweron(bool shouldDisableBatteryLoading);

  private:
    FCEU* fceu;

    unsigned int lagCounter;
    char lagFlag;
    bool lagCounterDisplay;
    bool frameAdvanceLagSkip;
    bool movieSubtitles;

    char curMovieFilename[512] = {0};
    MovieData currMovieData;
    MovieData defaultMovieData;
    int currRerecordCount; // Keep the global value

    bool subtitlesOnAVI = false;
    bool autoMovieBackup = false; //Toggle that determines if movies should be backed up automatically before altering them
    bool freshMovie = false;	  //True when a movie loads, false when movie is altered.  Used to determine if a movie has been altered since opening
    bool movieFromPoweron = true;

    int currFrameCounter;
    uint32 cur_input_display = 0;
    int pauseframe = -1;
    bool movie_readonly = true;
    int input_display = 0;
    int frame_display = 0;
    int rerecord_display = 0;
    bool fullSaveStateLoads = false;	//Option for loading a savestates full contents in read+write mode instead of up to the frame count in the savestate (useful as a recovery option)

    int _currCommand = 0;

    bool suppressMovieStop=false;

    EMOVIEMODE movieMode = MOVIEMODE_INACTIVE;

    EMUFILE* osRecordingMovie = NULL;

    SFORMAT FCEUMOV_STATEINFO[2]={
	    { &currFrameCounter, 4|FCEUSTATE_RLSB, "FCNT"},
	    { 0 }
    };

    char lagcounterbuf[32] = {0};

    std::vector<int> subtitleFrames;		//Frame numbers for subtitle messages
    std::vector<std::string> subtitleMessages;	//Messages of subtitles

    bool load_successful;

    // Methods.
    bool LoadFM2(MovieData& movieData, EMUFILE* fp, int size, bool stopAfterHeader);
    void LoadFM2_binarychunk(MovieData& movieData, EMUFILE* fp, int size);

    void StopPlayback();
    void FinishPlayback();
    void closeRecordingMovie();
    void StopRecording();
    void openRecordingMovie(const char* fname);

    int CheckTimelines(MovieData& stateMovie, MovieData& currMovie);
};

} // namespace fceu

#endif // define _MOVIE_H_
