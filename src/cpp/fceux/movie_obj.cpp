#include "movie_obj.h"

#include <zlib.h>

#include "fceu_obj.h"
#include "input_obj.h"
#include "netplay_obj.h"

namespace fceu {

void MovieData::clearRecordRange(int start, int len)
{
	for(int i=0;i<len;i++)
	{
		records[i+start].clear();
	}
}

void MovieData::eraseRecords(int at, int frames)
{
	if (at < (int)records.size())
	{
		if (frames == 1)
		{
			// erase 1 frame
			records.erase(records.begin() + at);
		} else
		{
			// erase many frames
			if (at + frames > (int)records.size())
				frames = (int)records.size() - at;
			records.erase(records.begin() + at, records.begin() + (at + frames));
		}
	}
}

void MovieData::insertEmpty(int at, int frames)
{
	if (at == -1)
	{
		records.resize(records.size() + frames);
	} else
	{
		records.insert(records.begin() + at, frames, MovieRecord());
	}
}

void MovieData::cloneRegion(int at, int frames)
{
	if (at < 0) return;

	records.insert(records.begin() + at, frames, MovieRecord());

	for(int i = 0; i < frames; i++)
		records[i + at].Clone(records[i + at + frames]);
}

bool MovieData::loadSavestateFrom(std::vector<uint8>* buf)
{
	EMUFILE_MEMORY ms(buf);
	return FCEUSS_LoadFP(&ms,SSLOADPARAM_BACKUP);
}

void MovieData::dumpSavestateTo(std::vector<uint8>* buf, int compressionLevel)
{
	EMUFILE_MEMORY ms(buf);
	FCEUSS_SaveMS(&ms,compressionLevel);
	ms.trim();
}
// ----------------------------------------------------------------------------
MovieRecord::MovieRecord()
{
	commands = 0;
	*(uint32*)&joysticks = 0;
	memset(zappers, 0, sizeof(zappers));
}

void MovieRecord::clear()
{
	commands = 0;
	*(uint32*)&joysticks = 0;
	memset(zappers, 0, sizeof(zappers));
}

bool MovieRecord::Compare(MovieRecord& compareRec)
{
	//Joysticks, Zappers, and commands

	if (this->commands != compareRec.commands)
		return false;
	if ((*(uint32*)&(this->joysticks)) != (*(uint32*)&(compareRec.joysticks)))
		return false;
	if (memcmp(this->zappers, compareRec.zappers, sizeof(zappers)))
		return false;

	/*
	if (this->joysticks != compareRec.joysticks)
		return false;

	//if new commands are ever recordable, they need to be added here if we go with this method
	if(this->command_reset() != compareRec.command_reset()) return false;
	if(this->command_power() != compareRec.command_power()) return false;
	if(this->command_fds_insert() != compareRec.command_fds_insert()) return false;
	if(this->command_fds_select() != compareRec.command_fds_select()) return false;

	if (this->zappers[0].x != compareRec.zappers[0].x) return false;
	if (this->zappers[0].y != compareRec.zappers[0].y) return false;
	if (this->zappers[0].zaphit != compareRec.zappers[0].zaphit) return false;
	if (this->zappers[0].b != compareRec.zappers[0].b) return false;
	if (this->zappers[0].bogo != compareRec.zappers[0].bogo) return false;

	if (this->zappers[1].x != compareRec.zappers[1].x) return false;
	if (this->zappers[1].y != compareRec.zappers[1].y) return false;
	if (this->zappers[1].zaphit != compareRec.zappers[1].zaphit) return false;
	if (this->zappers[1].b != compareRec.zappers[1].b) return false;
	if (this->zappers[1].bogo != compareRec.zappers[1].bogo) return false;
	*/

	return true;
}
void MovieRecord::Clone(MovieRecord& sourceRec)
{
	*(uint32*)&joysticks = *(uint32*)(&(sourceRec.joysticks));
	memcpy(this->zappers, sourceRec.zappers, sizeof(zappers));
	this->commands = sourceRec.commands;
}

const char MovieRecord::mnemonics[8] = {'A','B','S','T','U','D','L','R'};

void MovieRecord::dumpJoy(EMUFILE* os, uint8 joystate)
{
	//these are mnemonics for each joystick bit.
	//since we usually use the regular joypad, these will be more helpful.
	//but any character other than ' ' or '.' should count as a set bit
	//maybe other input types will need to be encoded another way..
	for(int bit=7;bit>=0;bit--)
	{
		int bitmask = (1<<bit);
		char mnemonic = mnemonics[bit];
		//if the bit is set write the mnemonic
		if(joystate & bitmask)
			os->fwrite(&mnemonic,1);
		else //otherwise write an unset bit
			write8le('.',os);
	}
}

void MovieRecord::parseJoy(EMUFILE* is, uint8& joystate)
{
	char buf[8];
	is->fread(buf,8);
	joystate = 0;
	for(int i=0;i<8;i++)
	{
		joystate <<= 1;
		joystate |= ((buf[i]=='.'||buf[i]==' ')?0:1);
	}
}

void MovieRecord::parse(MovieData* md, EMUFILE* is)
{
	//by the time we get in here, the initial pipe has already been extracted

	//extract the commands
	commands = uint32DecFromIstream(is);
	//*is >> commands;
	is->fgetc(); //eat the pipe

	//a special case: if fourscore is enabled, parse four gamepads
	if(md->fourscore)
	{
		parseJoy(is,joysticks[0]); is->fgetc(); //eat the pipe
		parseJoy(is,joysticks[1]); is->fgetc(); //eat the pipe
		parseJoy(is,joysticks[2]); is->fgetc(); //eat the pipe
		parseJoy(is,joysticks[3]); is->fgetc(); //eat the pipe
	}
	else
	{
		for(int port=0;port<2;port++)
		{
			if(md->ports[port] == SI_GAMEPAD)
				parseJoy(is, joysticks[port]);

			is->fgetc(); //eat the pipe
		}
	}

	//(no fcexp data is logged right now)
	is->fgetc(); //eat the pipe

	//should be left at a newline
}

bool MovieRecord::parseBinary(MovieData* md, EMUFILE* is)
{
	commands = (uint8)is->fgetc();

	//check for eof
	if(is->eof()) return false;

	if(md->fourscore)
	{
		is->fread((char*)&joysticks,4);
	}
	else
	{
		for(int port=0;port<2;port++)
		{
			if(md->ports[port] == SI_GAMEPAD)
				joysticks[port] = (uint8)is->fgetc();
		}
	}

	return true;
}


void MovieRecord::dumpBinary(MovieData* md, EMUFILE* os, int index)
{
	write8le(commands,os);
	if(md->fourscore)
	{
		for(int i=0;i<4;i++)
			os->fwrite(&joysticks[i],sizeof(joysticks[i]));
	}
	else
	{
		for(int port=0;port<2;port++)
		{
			if(md->ports[port] == SI_GAMEPAD)
				os->fwrite(&joysticks[port],sizeof(joysticks[port]));
		}
	}
}

void MovieRecord::dump(MovieData* md, EMUFILE* os, int index)
{
	// dump the commands
	//*os << '|' << setw(1) << (int)commands;
	os->fputc('|');
	putdec<uint8,3,false>(os, commands);	// "variable length decimal integer"

	//a special case: if fourscore is enabled, dump four gamepads
	if(md->fourscore)
	{
		os->fputc('|');
		dumpJoy(os,joysticks[0]); os->fputc('|');
		dumpJoy(os,joysticks[1]); os->fputc('|');
		dumpJoy(os,joysticks[2]); os->fputc('|');
		dumpJoy(os,joysticks[3]); os->fputc('|');
	}
	else
	{
		for(int port=0;port<2;port++)
		{
			os->fputc('|');
			if(md->ports[port] == SI_GAMEPAD)
				dumpJoy(os, joysticks[port]);
		}
		os->fputc('|');
	}

	//(no fcexp data is logged right now)
	os->fputc('|');

	//each frame is on a new line
	os->fputc('\n');
}

MovieData::MovieData()
	: version(MOVIE_VERSION)
	, emuVersion(FCEU_VERSION_NUMERIC)
	, fds(false)
	, palFlag(false)
	, PPUflag(false)
	, rerecordCount(0)
	, binaryFlag(false)
	, loadFrameCount(-1)
	, microphone(false)
{
	memset(&romChecksum,0,sizeof(MD5DATA));
}

void MovieData::truncateAt(int frame)
{
	records.resize(frame);
}

void MovieData::installValue(std::string& key, std::string& val)
{
	//todo - use another config system, or drive this from a little data structure. because this is gross
	if(key == "FDS")
		installInt(val,fds);
	else if(key == "NewPPU")
		installBool(val,PPUflag);
	else if(key == "version")
		installInt(val,version);
	else if(key == "emuVersion")
		installInt(val,emuVersion);
	else if(key == "rerecordCount")
		installInt(val,rerecordCount);
	else if(key == "palFlag")
		installBool(val,palFlag);
	else if(key == "romFilename")
		romFilename = val;
	else if(key == "romChecksum")
		StringToBytes(val,&romChecksum,MD5DATA::size);
	else if(key == "guid")
		guid = Guid::fromString(val);
	else if(key == "fourscore")
		installBool(val,fourscore);
	else if(key == "microphone")
		installBool(val,microphone);
	else if(key == "port0")
		installInt(val,ports[0]);
	else if(key == "port1")
		installInt(val,ports[1]);
	else if(key == "binary")
		installBool(val,binaryFlag);
	else if(key == "comment")
		comments.push_back(mbstowcs(val));
	else if (key == "subtitle")
		subtitles.push_back(val); //mbstowcs(val));
	else if(key == "savestate")
	{
		int len = Base64StringToBytesLength(val);
		if(len == -1) len = HexStringToBytesLength(val); // wasn't base64, try hex
		if(len >= 1)
		{
			savestate.resize(len);
			StringToBytes(val,&savestate[0],len); // decodes either base64 or hex
		}
	}
	else if (key == "length")
	{
		installInt(val, loadFrameCount);
	}
}

int MovieData::dump(EMUFILE *os, bool binary)
{
	int start = os->ftell();
	os->fprintf("version %d\n", version);
	os->fprintf("emuVersion %d\n", emuVersion);
	os->fprintf("rerecordCount %d\n", rerecordCount);
	os->fprintf("palFlag %d\n" , (palFlag?1:0) );
	os->fprintf("romFilename %s\n" , romFilename.c_str() );
	os->fprintf("romChecksum %s\n" , BytesToString(romChecksum.data,MD5DATA::size).c_str() );
	os->fprintf("guid %s\n" , guid.toString().c_str() );
	os->fprintf("fourscore %d\n" , (fourscore?1:0) );
	os->fprintf("microphone %d\n" , (microphone?1:0) );
	os->fprintf("port0 %d\n" , ports[0] );
	os->fprintf("port1 %d\n" , ports[1] );
	os->fprintf("FDS %d\n" , fds?1:0 );
	os->fprintf("NewPPU %d\n" , PPUflag?1:0 );

	for(uint32 i=0;i<comments.size();i++)
		os->fprintf("comment %s\n" , wcstombs(comments[i]).c_str() );

	for(uint32 i=0;i<subtitles.size();i++)
		os->fprintf("subtitle %s\n" , subtitles[i].c_str() );

	if(binary)
		os->fprintf("binary 1\n" );

	if(savestate.size())
		os->fprintf("savestate %s\n" , BytesToString(&savestate[0],savestate.size()).c_str() );

	if (this->loadFrameCount >= 0)
		os->fprintf("length %d\n" , this->loadFrameCount);

	if(binary)
	{
		//put one | to start the binary dump
		os->fputc('|');
		for(int i=0;i<(int)records.size();i++)
			records[i].dumpBinary(this, os, i);
	} else
	{
		for(int i=0;i<(int)records.size();i++)
			records[i].dump(this, os, i);
	}

	int end = os->ftell();
	return end-start;
}

int Movie::GetFrame(void)
{
	return currFrameCounter;
}

int Movie::GetLagCount(void)
{
	return lagCounter;
}

void Movie::IncreaseLagCount(void)
{
	lagCounter++;
}

bool Movie::GetLagged(void)
{
	if (lagFlag)
		return true;
	else
		return false;
}
void Movie::SetLagFlag(bool value)
{
	lagFlag = (value) ? 1 : 0;
}

void Movie::LagCounterReset()
{
	lagCounter = 0;
}

void Movie::LagCounterToggle(void)
{
	lagCounterDisplay ^= 1;
}

void Movie::FA_SkipLag(void)
{
	frameAdvanceLagSkip ^= 1;
}

bool Movie::GetFrameAdvanceLagSkip(void) {
    return frameAdvanceLagSkip;
}

void Movie::SubtitleToggle(void)
{
	movieSubtitles ^= 1;
	if (movieSubtitles)	fceu::DispMessage("Movie subtitles on",0);
	else fceu::DispMessage("Movie subtitles off",0);
}

bool Movie::ShouldPause(void)
{
	if(pauseframe && currFrameCounter+1 == pauseframe)
	{
		pauseframe = 0;
		return true;
	}
	else
	{
		return false;
	}
}

EMOVIEMODE Movie::Mode()
{
	return movieMode;
}

bool Movie::Mode(EMOVIEMODE modemask)
{
	return (movieMode&modemask)!=0;
}

bool Movie::Mode(int modemask)
{
	return Mode((EMOVIEMODE)modemask);
}

void Movie::LoadFM2_binarychunk(MovieData& movieData, EMUFILE* fp, int size)
{
	int recordsize = 1; //1 for the command
	if(movieData.fourscore)
		recordsize += 4; //4 joysticks
	else
	{
		for(int i=0;i<2;i++)
		{
			switch(movieData.ports[i])
			{
			case SI_GAMEPAD: recordsize++; break;
			}
		}
	}

	//find out how much remains in the file
	int curr = fp->ftell();
	fp->fseek(0,SEEK_END);
	int end = fp->ftell();
	int flen = end-curr;
	fp->fseek(curr,SEEK_SET);

	//the amount todo is the min of the limiting size we received and the remaining contents of the file
	int todo = std::min(size, flen);

	int numRecords = todo/recordsize;
	if (movieData.loadFrameCount!=-1 && movieData.loadFrameCount<numRecords)
		numRecords=movieData.loadFrameCount;

	movieData.records.resize(numRecords);
	for(int i=0;i<numRecords;i++)
	{
		movieData.records[i].parseBinary(&movieData,fp);
	}
}

//yuck... another custom text parser.
bool Movie::LoadFM2(MovieData& movieData, EMUFILE* fp, int size, bool stopAfterHeader)
{
	// if there's no "binary" tag in the movie header, consider it as a movie in text format
	movieData.binaryFlag = false;
	// Non-TASEditor projects consume until EOF
	movieData.loadFrameCount = -1;

	std::ios::pos_type curr = fp->ftell();

	if (!stopAfterHeader)
	{
		// first, look for an fcm signature
		char fcmbuf[3];
		fp->fread(fcmbuf,3);
		fp->fseek(curr,SEEK_SET);
		if(!strncmp(fcmbuf,"FCM",3)) {
            fceu::PrintError("FCM File format is no longer supported. Please use Tools > Convert FCM");
			return false;
		}
	}

	//movie must start with "version 3"
	char buf[9];
	curr = fp->ftell();
	fp->fread(buf,9);
	fp->fseek(curr,SEEK_SET);
	if(fp->fail()) return false;
	if(memcmp(buf,"version 3",9))
		return false;

	std::string key,value;
	enum {
		NEWLINE, KEY, SEPARATOR, VALUE, RECORD, COMMENT, SUBTITLE
	} state = NEWLINE;
	bool bail = false;
	bool iswhitespace, isrecchar, isnewline;
	int c;
	for(;;)
	{
		if(size--<=0) goto bail;
		c = fp->fgetc();
		if(c == -1)
			goto bail;
		iswhitespace = (c==' '||c=='\t');
		isrecchar = (c=='|');
		isnewline = (c==10||c==13);
		if(isrecchar && movieData.binaryFlag && !stopAfterHeader)
		{
			LoadFM2_binarychunk(movieData, fp, size);
			return true;
		} else if (isnewline && movieData.loadFrameCount == movieData.records.size())
			// exit prematurely if loaded the specified amound of records
			return true;
		switch(state)
		{
		case NEWLINE:
			if(isnewline) goto done;
			if(iswhitespace) goto done;
			if(isrecchar)
				goto dorecord;
			//must be a key
			key = "";
			value = "";
			goto dokey;
			break;
		case RECORD:
			{
				dorecord:
				if (stopAfterHeader) return true;
				int currcount = movieData.records.size();
				movieData.records.resize(currcount+1);
				int preparse = fp->ftell();
				movieData.records[currcount].parse(&movieData, fp);
				int postparse = fp->ftell();
				size -= (postparse-preparse);
				state = NEWLINE;
				break;
			}

		case KEY:
			dokey: //dookie
			state = KEY;
			if(iswhitespace) goto doseparator;
			if(isnewline) goto commit;
			key += c;
			break;
		case SEPARATOR:
			doseparator:
			state = SEPARATOR;
			if(isnewline) goto commit;
			if(!iswhitespace) goto dovalue;
			break;
		case VALUE:
			dovalue:
			state = VALUE;
			if(isnewline) goto commit;
			value += c;
		}
		goto done;

		bail:
		bail = true;
		if(state == VALUE) goto commit;
		goto done;
		commit:
		movieData.installValue(key,value);
		state = NEWLINE;
		done: ;
		if(bail) break;
	}

	return true;
}

/// Stop movie playback.
void Movie::StopPlayback()
{
    fceu::DispMessageOnMovie("Movie playback stopped.");
	movieMode = MOVIEMODE_INACTIVE;
}

// Stop movie playback without closing the movie.
void Movie::FinishPlayback()
{
    fceu::DispMessage("Movie finished playing.",0);
	movieMode = MOVIEMODE_FINISHED;
}

void Movie::closeRecordingMovie()
{
	if(osRecordingMovie)
	{
		delete osRecordingMovie;
		osRecordingMovie = 0;
	}
}

/// Stop movie recording
void Movie::StopRecording()
{
    fceu::DispMessage("Movie recording stopped.",0);
	movieMode = MOVIEMODE_INACTIVE;

	closeRecordingMovie();
}

void Movie::Stop()
{
	if(suppressMovieStop)
		return;

	if(movieMode == MOVIEMODE_PLAY || movieMode == MOVIEMODE_FINISHED)
		StopPlayback();
	else if(movieMode == MOVIEMODE_RECORD)
		StopRecording();

	curMovieFilename[0] = 0;			//No longer a current movie filename
	freshMovie = false;					//No longer a fresh movie loaded
	if (bindSavestate) fceu->SetAutoSS(false);	//If bind movies to savestates is true, then there is no longer a valid auto-save to load

}

void Movie::poweron(bool shouldDisableBatteryLoading)
{
	//// make a for-movie-recording power-on clear the game's save data, too
	//extern char lastLoadedGameName [2048];
	//extern int disableBatteryLoading, suppressAddPowerCommand;
	//suppressAddPowerCommand=1;
	//if(shouldDisableBatteryLoading) disableBatteryLoading=1;
	//suppressMovieStop=true;
	//{
	//	//we need to save the pause state through this process
	//	int oldPaused = EmulationPaused;

	//	// NOTE:  this will NOT write an FCEUNPCMD_POWER into the movie file
	//	FCEUGI* gi = FCEUI_LoadGame(lastLoadedGameName, 0);
	//	assert(gi);
	//	PowerNES();

	//	EmulationPaused = oldPaused;
	//}
	//suppressMovieStop=false;
	//if(shouldDisableBatteryLoading) disableBatteryLoading=0;
	//suppressAddPowerCommand=0;

	fceu->cart.SetDisableBatteryLoading(1);
	fceu->PowerNES();
	fceu->cart.SetDisableBatteryLoading(0);
}

void Movie::CreateCleanMovie()
{
	currMovieData = MovieData();
	currMovieData.palFlag = fceu->GetCurrentVidSystem(0,0)!=0;
	currMovieData.romFilename = FileBase;
	currMovieData.romChecksum = fceu->GetGameInfo()->MD5;
	currMovieData.guid.newGuid();
	currMovieData.fourscore = fceu->input.GetFourscore();
	currMovieData.microphone = 0;
	currMovieData.ports[0] = fceu->input.GetJoyport(0)->type;
	currMovieData.ports[1] = fceu->input.GetJoyport(1)->type;
	currMovieData.fds = isFDS;
	currMovieData.PPUflag = (fceu->ppu.is_newppu() != 0);
}
void Movie::ClearCommands()
{
	_currCommand = 0;
}

bool Movie::FromPoweron()
{
	return movieFromPoweron;
}

//begin playing an existing movie
bool Movie::Load(const char *fname, bool _read_only, int _pauseframe)
{
	if(!fceu->IsValidUI(FCEUI_PLAYMOVIE))
		return true;	//adelikat: file did not fail to load, so let's return true here, just do nothing

	assert(fname);

	//mbg 6/10/08 - we used to call StopMovie here, but that cleared curMovieFilename and gave us crashes...
	if(movieMode == MOVIEMODE_PLAY || movieMode == MOVIEMODE_FINISHED)
		StopPlayback();
	else if(movieMode == MOVIEMODE_RECORD)
		StopRecording();
	//--------------

	currMovieData = MovieData();

	strcpy(curMovieFilename, fname);
	FCEUFILE *fp = fceu::fopen(fname,0,"rb",0);
	if (!fp) return false;
	if(fp->isArchive() && !_read_only) {
        fceu::PrintError("Cannot open a movie in read+write from an archive.");
		return true;	//adelikat: file did not fail to load, so return true (false is only for file not exist/unable to open errors
	}

	LoadFM2(currMovieData, fp->stream, fp->size, false);
	LoadSubtitles(currMovieData);
	delete fp;

	freshMovie = true;	//Movie has been loaded, so it must be unaltered
	if (bindSavestate) fceu->SetAutoSS(false);	//If bind savestate to movie is true, then their isn't a valid auto-save to load, so flag it
	//fully reload the game to reinitialize everything before playing any movie
	poweron(true);

	if(currMovieData.savestate.size())
	{
		//WE NEED TO LOAD A SAVESTATE
		movieFromPoweron = false;
		bool success = MovieData::loadSavestateFrom(&currMovieData.savestate);
		if(!success) return true;	//adelikat: I guess return true here?  False is only for a bad movie filename, if it got this far the file was good?
	} else {
		movieFromPoweron = true;
	}

	//if there is no savestate, we won't have this crucial piece of information at the start of the movie.
	//so, we have to include it with the movie
	if(currMovieData.palFlag)
		fceu->SetVidSystem(1);
	else
		fceu->SetVidSystem(0);

	//stuff that should only happen when we're ready to positively commit to the replay
	currFrameCounter = 0;
	pauseframe = _pauseframe;
	movie_readonly = _read_only;
	movieMode = MOVIEMODE_PLAY;
	if (movieMode != MOVIEMODE_TASEDITOR)
		currRerecordCount = currMovieData.rerecordCount;

	if(movie_readonly)
		fceu::DispMessage("Replay started Read-Only.",0);
	else
		fceu::DispMessage("Replay started Read+Write.",0);

	#ifdef CREATE_AVI
	if(LoggingEnabled)
	{
        fceu::DispMessage("Video recording enabled.\n",0);
	    LoggingEnabled = 2;
	}
	#endif

	return true;
}

void Movie::openRecordingMovie(const char* fname)
{
	osRecordingMovie = EMUFILE_FILE::FCEUD_UTF8_fstream(fname, "wb");
	if(!osRecordingMovie)
		fceu::PrintError("Error opening movie output file: %s",fname);
	strcpy(curMovieFilename, fname);
}

//begin recording a new movie
//TODO - BUG - the record-from-another-savestate doesnt work.
void Movie::Save(const char *fname, EMOVIE_FLAG flags, std::wstring author)
{
	if(!fceu->IsValidUI(FCEUI_RECORDMOVIE))
		return;

	assert(fname);

	Stop();

	openRecordingMovie(fname);

	currFrameCounter = 0;
	LagCounterReset();
	CreateCleanMovie();
	if(author != L"") currMovieData.comments.push_back(L"author " + author);

	if(flags & MOVIE_FLAG_FROM_POWERON)
	{
		movieFromPoweron = true;
		poweron(true);
	}
	else
	{
		movieFromPoweron = false;
		MovieData::dumpSavestateTo(&currMovieData.savestate,Z_BEST_COMPRESSION);
	}

	ClearCommands();

	//we are going to go ahead and dump the header. from now on we will only be appending frames
	currMovieData.dump(osRecordingMovie, false);

	movieMode = MOVIEMODE_RECORD;
	movie_readonly = false;
	if (movieMode != MOVIEMODE_TASEDITOR)
		currRerecordCount = 0;

    fceu::DispMessage("Movie recording started.",0);
}

//the main interaction point between the emulator and the movie system.
//either dumps the current joystick state or loads one state from the movie
void Movie::AddInputState()
{
	if (movieMode == MOVIEMODE_PLAY)
	{
		//stop when we run out of frames
		if (currFrameCounter >= (int)currMovieData.records.size())
		{
			FinishPlayback();
			//tell all drivers to poll input and set up their logical states
			for(int port=0;port<2;port++) {
                JOYPORT* joyport = fceu->input.GetJoyport(port);
				joyport->driver->Update(port,joyport->ptr,joyport->attrib);
            }
		} else
		{
			MovieRecord* mr = &currMovieData.records[currFrameCounter];

			//reset and power cycle if necessary
			if(mr->command_power())
				fceu->PowerNES();
			if(mr->command_reset())
				fceu->ResetNES();
			if(mr->command_fds_insert())
				FCEU_FDSInsert();
			if(mr->command_fds_select())
				FCEU_FDSSelect();
			if (mr->command_vs_insertcoin())
				FCEU_VSUniCoin();

			fceu->input.GetJoyport(0)->load(mr);
			fceu->input.GetJoyport(1)->load(mr);
		}

		//pause the movie at a specified frame
		if (ShouldPause() && fceu->EmulationPaused()==0)
		{
			fceu->ToggleEmulationPause();
            fceu::DispMessage("Paused at specified movie frame",0);
		}

	} else if (movieMode == MOVIEMODE_RECORD)
	{
		MovieRecord mr;

		fceu->input.GetJoyport(0)->log(&mr);
		fceu->input.GetJoyport(1)->log(&mr);
		mr.commands = _currCommand;
		_currCommand = 0;

		//Adelikat: in normal mode, this is done at the time of loading a savestate in read+write mode
		//If the user chooses it can be delayed to here
		if (fullSaveStateLoads && (currFrameCounter < (int)currMovieData.records.size()))
			currMovieData.truncateAt(currFrameCounter);

		mr.dump(&currMovieData, osRecordingMovie,currMovieData.records.size());	// to disk

		currMovieData.records.push_back(mr);
	}

	currFrameCounter++;

	fceu->input.UpdateJoy(cur_input_display);
}

//TODO
void Movie::AddCommand(int cmd)
{
	// do nothing if not recording a movie
	if(movieMode != MOVIEMODE_RECORD && movieMode != MOVIEMODE_TASEDITOR)
		return;

	// translate "FCEU NetPlay" command to "FCEU Movie" command
	switch (cmd)
	{
		case FCEUNPCMD_RESET: cmd = MOVIECMD_RESET; break;
		case FCEUNPCMD_POWER: cmd = MOVIECMD_POWER; break;
		case FCEUNPCMD_FDSINSERT: cmd = MOVIECMD_FDS_INSERT; break;
		case FCEUNPCMD_FDSSELECT: cmd = MOVIECMD_FDS_SELECT; break;
		case FCEUNPCMD_VSUNICOIN: cmd = MOVIECMD_VS_INSERTCOIN; break;
		// all other netplay commands (e.g. FCEUNPCMD_VSUNIDIP0) are not supported by movie recorder for now
		default: return;
	}

	_currCommand |= cmd;
}

void Movie::DrawMovies(uint8 *XBuf)
{
	if(frame_display)
	{
		char counterbuf[32] = {0};
		int color = 0x20;
		if(movieMode == MOVIEMODE_PLAY)
			sprintf(counterbuf,"%d/%d",currFrameCounter,(int)currMovieData.records.size());
		else if(movieMode == MOVIEMODE_RECORD)
			sprintf(counterbuf,"%d",currFrameCounter);
		else if (movieMode == MOVIEMODE_FINISHED)
		{
			sprintf(counterbuf,"%d/%d (finished)",currFrameCounter,(int)currMovieData.records.size());
			color = 0x17; //Show red to get attention
		} else if(movieMode == MOVIEMODE_TASEDITOR)
		{
			sprintf(counterbuf,"%d",currFrameCounter);
		} else
			sprintf(counterbuf,"%d (no movie)",currFrameCounter);

		if(counterbuf[0])
			fceu->drawing.DrawTextTrans(ClipSidesOffset+XBuf+fceu->TextScanlineOffsetFromBottom(30)+1, 256, (uint8*)counterbuf, color+0x80);
	}
	if(rerecord_display && movieMode != MOVIEMODE_INACTIVE)
	{
		char counterbuf[32] = {0};
		sprintf(counterbuf,"%d",currMovieData.rerecordCount);

		if(counterbuf[0])
			fceu->drawing.DrawTextTrans(ClipSidesOffset+XBuf+fceu->TextScanlineOffsetFromBottom(50)+1, 256, (uint8*)counterbuf, 0x28+0x80);
	}
}

void Movie::DrawLagCounter(uint8 *XBuf)
{
	if (lagCounterDisplay)
	{
		// If currently lagging - display red, else display green
		uint8 color = (lagFlag) ? (0x16+0x80) : (0x2A+0x80);
		sprintf(lagcounterbuf, "%d", lagCounter);
		if(lagcounterbuf[0])
			fceu->drawing.DrawTextTrans(ClipSidesOffset + XBuf + fceu->TextScanlineOffsetFromBottom(40) + 1, 256, (uint8*)lagcounterbuf, color);
	}
}

int Movie::WriteState(EMUFILE* os)
{
	//we are supposed to dump the movie data into the savestate
	if(movieMode == MOVIEMODE_RECORD || movieMode == MOVIEMODE_PLAY || movieMode == MOVIEMODE_FINISHED)
		return currMovieData.dump(os, true);
	else return 0;
}

// returns
int Movie::CheckTimelines(MovieData& stateMovie, MovieData& currMovie)
{
	// end_frame = min(urrMovie.records.size(), stateMovie.records.size(), currFrameCounter)
	int end_frame = currMovie.records.size();
	if (end_frame > (int)stateMovie.records.size())
		end_frame = stateMovie.records.size();
	if (end_frame > currFrameCounter)
		end_frame = currFrameCounter;

	for (int x = 0; x < end_frame; x++)
	{
		if (!stateMovie.records[x].Compare(currMovie.records[x]))
			return x;
	}
	// no mismatch found
	return -1;
}

bool Movie::ReadState(EMUFILE* is, uint32 size)
{
	load_successful = false;

	if (!movie_readonly)
	{
		if (currMovieData.loadFrameCount >= 0)
		{
            fceu::printf("This movie is a TAS Editor project file! It can be modified in TAS Editor only.\nMovie is now Read-Only.\n");
			movie_readonly = true;
		}
		if (fceu::isFileInArchive(curMovieFilename))
		{
			//a little rule: cant load states in read+write mode with a movie from an archive.
			//so we are going to switch it to readonly mode in that case
            fceu::PrintError("Cannot loadstate in Read+Write with movie from archive. Movie is now Read-Only.");
			movie_readonly = true;
		}
	}

	MovieData tempMovieData = MovieData();
	std::ios::pos_type curr = is->ftell();
	if(!LoadFM2(tempMovieData, is, size, false)) {
		is->fseek((uint32)curr+size,SEEK_SET);
		if(FCEU_state_loading_old_format) {
			if(movieMode == MOVIEMODE_PLAY || movieMode == MOVIEMODE_RECORD || movieMode == MOVIEMODE_FINISHED) {
				//FCEUI_StopMovie();  //No reason to stop the movie, nothing destructive has happened yet.
                fceu::PrintError("You have tried to use an old savestate while playing a movie. This is unsupported (since the old savestate has old-format movie data in it which can't be converted on the fly)");
			}
		}
		return false;
	}

	//----------------
	//complex TAS logic for loadstate
	//fully conforms to the savestate logic documented in the Laws of TAS
	//http://tasvideos.org/LawsOfTAS/OnSavestates.html
	//----------------

	/*
	Playback or Recording + Read-only

    * Check that GUID of movie and savestate-movie must match or else error
          o on error: a message informing that the savestate doesn't belong to this movie. This is a GUID mismatch error. Give user a choice to load it anyway.
                + failstate: if use declines, loadstate attempt canceled, movie can resume as if not attempted if user has backup savstates enabled else stop movie
    * Check that movie and savestate-movie are in same timeline. If not then this is a wrong timeline error.
          o on error: a message informing that the savestate doesn't belong to this movie
                + failstate: loadstate attempt canceled, movie can resume as if not attempted if user has backup savestates enabled else stop movie
    * Check that savestate-movie is not greater than movie. If it's greater then this is a future event error and is not allowed in read-only
          o on error: message informing that the savestate is from a frame after the last frame of the movie
                + failstate - loadstate attempt cancelled, movie can resume if user has backup savesattes enabled, else stop movie
    * Check that savestate framcount <= savestate movie length. If not this is a post-movie savestate and is not allowed in read-only
          o on error: message informing that the savestate is from a frame after the last frame of the savestated movie
                + failstate - loadstate attempt cancelled, movie can resume if user has backup savesattes enabled, else stop movie
    * All error checks have passed, state will be loaded
    * Movie contained in the savestate will be discarded
    * Movie is now in Playback mode

	Playback, Recording + Read+write

    * Check that GUID of movie and savestate-movie must match or else error
          o on error: a message informing that the savestate doesn't belong to this movie. This is a GUID mismatch error. Give user a choice to load it anyway.
                + failstate: if use declines, loadstate attempt canceled, movie can resume as if not attempted (stop movie if resume fails)canceled, movie can resume if backup savestates enabled else stopmovie
    * Check that savestate framcount <= savestate movie length. If not this is a post-movie savestate
          o on post-movie: See post-movie event section.
    * savestate passed all error checks and will now be loaded in its entirety and replace movie (note: there will be no truncation yet)
    * current framecount will be set to savestate framecount
    * on the next frame of input, movie will be truncated to framecount
          o (note: savestate-movie can be a future event of the movie timeline, or a completely new timeline and it is still allowed)
    * Rerecord count of movie will be incremented
    * Movie is now in record mode

	Post-movie savestate event

    * Whan a savestate is loaded and is determined that the savestate-movie length is less than the savestate framecount then it is a post-movie savestate. These occur when a savestate was made during Movie Finished mode.
	* If read+write, the entire savestate movie will be loaded and replace current movie.
    * If read-only, the savestate movie will be discarded
    * Mode will be switched to Move Finished
    * Savestate will be loaded
    * Current framecount changes to savestate framecount
    * User will have control of input as if Movie inactive mode
	*/

	if(movieMode == MOVIEMODE_PLAY || movieMode == MOVIEMODE_RECORD || movieMode == MOVIEMODE_FINISHED)
	{
		//handle moviefile mismatch
		if(tempMovieData.guid != currMovieData.guid)
		{
			//mbg 8/18/08 - this code  can be used to turn the error message into an OK/CANCEL
				if (!backupSavestates) //If backups are disabled we can just resume normally since we can't restore so stop movie and inform user
				{
                    fceu::PrintError("Mismatch between savestate's movie and current movie.\ncurrent: %s\nsavestate: %s\nUnable to restore backup, movie playback stopped.\n",currMovieData.guid.toString().c_str(),tempMovieData.guid.toString().c_str());
					Stop();
				}
				else
				fceu::PrintError("Mismatch between savestate's movie and current movie.\ncurrent: %s\nsavestate: %s\n",currMovieData.guid.toString().c_str(),tempMovieData.guid.toString().c_str());

				return false;
		}

		closeRecordingMovie();

		if (movie_readonly)
		{
			// currFrameCounter at this point represents the savestate framecount
			int frame_of_mismatch = CheckTimelines(tempMovieData, currMovieData);
			if (frame_of_mismatch >= 0)
			{
				// Wrong timeline, do apprioriate logic here
				if (!backupSavestates)	//If backups are disabled we can just resume normally since we can't restore so stop movie and inform user
				{
                    fceu::PrintError("Error: Savestate not in the same timeline as movie!\nFrame %d branches from current timeline\nUnable to restore backup, movie playback stopped.", frame_of_mismatch);
					Stop();
				} else
					fceu::PrintError("Error: Savestate not in the same timeline as movie!\nFrame %d branches from current timeline", frame_of_mismatch);
				return false;
			} else if (movieMode == MOVIEMODE_FINISHED
				&& currFrameCounter > (int)currMovieData.records.size()
				&& currMovieData.records.size() == tempMovieData.records.size())
			{
				// special case (in MOVIEMODE_FINISHED mode)
				// allow loading post-movie savestates that were made after finishing current movie

			} else if (currFrameCounter > (int)currMovieData.records.size())
			{
				// this is future event state, don't allow it
				//TODO: turn frame counter to red to get attention
				if (!backupSavestates)	//If backups are disabled we can just resume normally since we can't restore so stop movie and inform user
				{
                    fceu::PrintError("Error: Savestate is from a frame (%d) after the final frame in the movie (%d). This is not permitted.\nUnable to restore backup, movie playback stopped.", currFrameCounter, currMovieData.records.size()-1);
					Stop();
				} else
					fceu::PrintError("Savestate is from a frame (%d) after the final frame in the movie (%d). This is not permitted.", currFrameCounter, currMovieData.records.size()-1);
				return false;
			} else if (currFrameCounter > (int)tempMovieData.records.size())
			{
				// this is post-movie savestate, don't allow it
				//TODO: turn frame counter to red to get attention
				if (!backupSavestates)	//If backups are disabled we can just resume normally since we can't restore so stop movie and inform user
				{
                    fceu::PrintError("Error: Savestate is from a frame (%d) after the final frame in the savestated movie (%d). This is not permitted.\nUnable to restore backup, movie playback stopped.", currFrameCounter, tempMovieData.records.size()-1);
					Stop();
				} else
					fceu::PrintError("Savestate is from a frame (%d) after the final frame in the savestated movie (%d). This is not permitted.", currFrameCounter, tempMovieData.records.size()-1);
				return false;
			} else
			{
				// Finally, this is a savestate file for this movie
				movieMode = MOVIEMODE_PLAY;
			}
		} else
		{
			//Read+Write mode
			if (currFrameCounter > (int)tempMovieData.records.size())
			{
				//This is a post movie savestate, handle it differently
				//Replace movie contents but then switch to movie finished mode
				currMovieData = tempMovieData;
				openRecordingMovie(curMovieFilename);
				currMovieData.dump(osRecordingMovie, false/*currMovieData.binaryFlag*/);
				FinishPlayback();
			} else
			{
				//truncate before we copy, just to save some time, unless the user selects a full copy option
				if (!fullSaveStateLoads)
					//we can only assume this here since we have checked that the frame counter is not greater than the movie data
					tempMovieData.truncateAt(currFrameCounter);
				
				currMovieData = tempMovieData;
				IncrementRerecordCount();
				openRecordingMovie(curMovieFilename);
				currMovieData.dump(osRecordingMovie, false/*currMovieData.binaryFlag*/);
				movieMode = MOVIEMODE_RECORD;

			}
		}
	}

	load_successful = true;

	return true;
}

void Movie::PreLoad(void)
{
	load_successful=0;
}

bool Movie::PostLoad(void)
{
	if(movieMode == MOVIEMODE_INACTIVE || movieMode == MOVIEMODE_TASEDITOR)
		return true;
	else
		return load_successful;
}

void Movie::IncrementRerecordCount()
{
	if (movieMode != MOVIEMODE_TASEDITOR)
		currRerecordCount++;
	else
		currMovieData.rerecordCount++;
	if (movieMode != MOVIEMODE_TASEDITOR)
		currMovieData.rerecordCount = currRerecordCount;
}

void Movie::ToggleFrameDisplay(void)
{
	frame_display=!frame_display;
}

void Movie::ToggleRerecordDisplay()
{
	rerecord_display ^= 1;
}

void Movie::ToggleInputDisplay(void)
{
	switch(input_display)
	{
	case 0:
		input_display = 1;
		break;
	case 1:
		input_display = 2;
		break;
	case 2:
		input_display = 4;
		break;
	default:
		input_display = 0;
		break;
	}
}

int Movie::GetLength()
{
	return currMovieData.records.size();
}

int Movie::GetRerecordCount()
{
	return currMovieData.rerecordCount;
}

bool Movie::GetToggleReadOnly()
{
	return movie_readonly;
}

void Movie::SetToggleReadOnly(bool which)
{
	if (which)	//If set to readonly
	{
		if (!movie_readonly)	//If not already set
		{
			movie_readonly = true;
            fceu::DispMessage("Movie is now Read-Only.",0);
		}
		else					//Else restate message
			fceu::DispMessage("Movie is Read-Only.",0);
	}
	else		//If set to read+write
	{
		if (movie_readonly)		//If not already set
		{
			movie_readonly = false;
            fceu::DispMessage("Movie is now Read+Write.",0);
		}
		else					//Else restate message
			fceu::DispMessage("Movie is Read+Write.",0);
	}
}
void Movie::ToggleReadOnly()
{
	char message[260];

	if(movie_readonly)
		strcpy(message, "Movie is now Read+Write");
	else
		strcpy(message, "Movie is now Read-Only");

	if (movieMode == MOVIEMODE_INACTIVE)
		strcat(message, " (no movie)");
	else if (movieMode == MOVIEMODE_FINISHED)
		strcat(message, " (finished)");

    fceu::DispMessage(message,0);
	movie_readonly = !movie_readonly;
}

void Movie::PlayFromBeginning(void)
{
	if (movieMode == MOVIEMODE_TASEDITOR)
	{
	} else if (movieMode != MOVIEMODE_INACTIVE)
	{
		if (currMovieData.savestate.empty())
		{
			movie_readonly = true;
			movieMode = MOVIEMODE_PLAY;
			poweron(true);
			currFrameCounter = 0;
            fceu::DispMessage("Movie is now Read-Only. Playing from beginning.",0);
		}
		else
		{
			// movie starting from savestate - reload movie file
            std::string str = curMovieFilename;
			Stop();
			if (Load(str.c_str(), 1, 0))
			{
				movieMode = MOVIEMODE_PLAY;
				movie_readonly = true;
                fceu::DispMessage("Movie is now Read-Only. Playing from beginning.",0);
			}
			//currMovieData.loadSavestateFrom(&currMovieData.savestate); //TODO: make something like this work instead so it doesn't have to reload
		}
	}
}

std::string Movie::GetName(void)
{
	return curMovieFilename;
}

bool Movie::GetInfo(FCEUFILE* fp, MOVIE_INFO& info, bool skipFrameCount)
{
	MovieData md;
	if(!LoadFM2(md, fp->stream, fp->size, skipFrameCount))
		return false;

	info.movie_version = md.version;
	info.poweron = md.savestate.size()==0;
	info.reset = false; //Soft-reset isn't used from starting movies anymore, so this will be false, better for FCEUFILE to have that info (as |1| on the first frame indicates it
	info.pal = md.palFlag;
	info.ppuflag = md.PPUflag;
	info.nosynchack = true;
	info.num_frames = md.records.size();
	info.md5_of_rom_used = md.romChecksum;
	info.emu_version_used = md.emuVersion;
	info.name_of_rom_used = md.romFilename;
	info.rerecord_count = md.rerecordCount;
	info.comments = md.comments;
	info.subtitles = md.subtitles;

	return true;
}

//This function creates an array of frame numbers and corresponding strings for displaying subtitles
void Movie::LoadSubtitles(MovieData &moviedata)
{
	subtitleFrames.resize(0);
	subtitleMessages.resize(0);
	for(uint32 i=0; i < moviedata.subtitles.size() ; i++)
	{
		std::string& subtitle = moviedata.subtitles[i];
		size_t splitat = subtitle.find_first_of(' ');
		std::string key, value;

		//If we can't split them, then don't process this one
		if(splitat == std::string::npos)
		{
		}
		//Else split the subtitle into the int and string arrays
		else
		{
			key = subtitle.substr(0,splitat);
			value = subtitle.substr(splitat+1);
			subtitleFrames.push_back(atoi(key.c_str()));
			subtitleMessages.push_back(value);
		}
	}

}

//Every frame, this will be called to determine if a subtitle should be displayed, which one, and then to display it
void Movie::ProcessSubtitles(void)
{
	if (movieMode == MOVIEMODE_INACTIVE) return;

	for(uint32 i=0;i<currMovieData.subtitles.size();i++)
	{
		if (currFrameCounter == subtitleFrames[i])
			DisplaySubtitles("%s",subtitleMessages[i].c_str());
	}
}

void Movie::DisplaySubtitles(char *format, ...)
{
	va_list ap;

	va_start(ap,format);
	vsnprintf(subtitleMessage.errmsg,sizeof(subtitleMessage.errmsg),format,ap);
	va_end(ap);

	subtitleMessage.howlong = 400;
	subtitleMessage.isMovieMessage = subtitlesOnAVI;
	subtitleMessage.linesFromBottom = 0;
}

void Movie::CreateFile(std::string fn)
{
	MovieData md = currMovieData;							//Get current movie data
	EMUFILE* outf = EMUFILE_FILE::FCEUD_UTF8_fstream(fn, "wb");		//open/create file
	md.dump(outf,false);									//dump movie data
	delete outf;											//clean up, delete file object
}

void Movie::MakeBackup(bool dispMessage)
{
	//This function generates backup movie files
    std::string currentFn;					//Current movie fillename
    std::string backupFn;					//Target backup filename
    std::string tempFn;						//temp used in back filename creation
    std::stringstream stream;
	int x;								//Temp variable for string manip
	bool exist = false;					//Used to test if filename exists
	bool overflow = false;				//Used for special situation when backup numbering exceeds limit

	currentFn = curMovieFilename;		//Get current moviefilename
	backupFn = curMovieFilename;		//Make backup filename the same as current moviefilename
	x = backupFn.find_last_of(".");		 //Find file extension
	backupFn = backupFn.substr(0,x);	//Remove extension
	tempFn = backupFn;					//Store the filename at this point
	for (unsigned int backNum=0;backNum<999;backNum++) //999 = arbituary limit to backup files
	{
		stream.str("");					 //Clear stream
		if (backNum > 99)
			stream << "-" << backNum;	 //assign backNum to stream
		else if (backNum <=99 && backNum >= 10)
			stream << "-0";				//Make it 010, etc if two digits
		else
			stream << "-00" << backNum;	 //Make it 001, etc if single digit
		backupFn.append(stream.str());	 //add number to bak filename
		backupFn.append(".bak");		 //add extension

		exist = CheckFileExists(backupFn.c_str());	//Check if file exists

		if (!exist)
			break;						//Yeah yeah, I should use a do loop or something
		else
		{
			backupFn = tempFn;			//Before we loop again, reset the filename

			if (backNum == 999)			//If 999 exists, we have overflowed, let's handle that
			{
				backupFn.append("-001.bak"); //We are going to simply overwrite 001.bak
				overflow = true;		//Flag that we have exceeded limit
				break;					//Just in case
			}
		}
	}
	CreateFile(backupFn);

	//TODO, decide if fstream successfully opened the file and print error message if it doesn't

	if (dispMessage)	//If we should inform the user
	{
		if (overflow)
			fceu::DispMessage("Backup overflow, overwriting %s",0,backupFn.c_str()); //Inform user of overflow
		else
			fceu::DispMessage("%s created",0,backupFn.c_str()); //Inform user of backup filename
	}
}


} // namespace fceu
