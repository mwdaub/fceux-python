#include "fceu_obj.h"

using namespace std;

namespace fceu {

void FCEU::TogglePPU(void) {
    ppu.TogglePPU();
    int newppu = ppu.is_newppu();
	if (newppu) {
        fceu::DispMessage("New PPU loaded", 0);
        fceu::printf("New PPU loaded");
		overclock_enabled = 0;
	} else {
        fceu::DispMessage("Old PPU loaded", 0);
        fceu::printf("Old PPU loaded");
	}
	normalscanlines = (dendy ? 290 : 240)+newppu; // use flag as number!
}

void FCEU::CloseGame(void) {
	if (GameInfo) {
		if (AutoResumePlay) {
			// save "-resume" savestate
			FCEUSS_Save(file.MakeFName(FCEUMKF_RESUMESTATE, 0, 0).c_str(), false);
		}

		if (GameInfo->name) {
            fceu::free(GameInfo->name);
			GameInfo->name = NULL;
		}

		if (GameInfo->type != GIT_NSF) {
            cheat.FlushGameCheats(0, 0);
		}

		(*GameInterface)(GI_CLOSE);

		movie.Stop();

		ResetExState(0, 0);

		//clear screen when game is closed
		if (XBuf)
			memset(XBuf, 0, 256 * 256);

		cart.CloseGenie();

		delete GameInfo;
		GameInfo = NULL;

		movie.SetFrame(0);

		//Reset flags for Undo/Redo/Auto Savestating //adelikat: TODO: maybe this stuff would be cleaner as a struct or class
		lastSavestateMade[0] = 0;
		undoSS = false;
		redoSS = false;
		lastLoadstateMade[0] = 0;
		undoLS = false;
		redoLS = false;
		AutoSS = false;
	}
}

void FCEU::ResetGameLoaded(void) {
	if (GameInfo) CloseGame();
	EmulationPaused_ = 0; //mbg 5/8/08 - loading games while paused was bad news. maybe this fixes it
	GameStateRestore = 0;
	ppu.ResetGameLoaded();
	if (GameExpSound.Kill)
		(*GameExpSound.Kill)();
	memset(&GameExpSound, 0, sizeof(GameExpSound));
	x6502.ResetGameLoaded();
	PAL &= 1;
	default_palette_selection = 0;
}

//name should be UTF-8, hopefully, or else there may be trouble
FCEUGI* FCEU::LoadGameVirtual(const char *name, int OverwriteVidMode, bool silent) {
	//----------
	//attempt to open the files
	FCEUFILE *fp;
	char fullname[2048];	// this name contains both archive name and ROM file name
	int lastpal = PAL;
	int lastdendy = dendy;

	const char* romextensions[] = { "nes", "fds", 0 };
	fp = fceu::fopen(name, 0, "rb", 0, -1, romextensions);

	if (!fp)
	{
		if (!silent)
			fceu::PrintError("Error opening \"%s\"!", name);
		return 0;
	} else if (fp->archiveFilename != "")
	{
		strcpy(fullname, fp->archiveFilename.c_str());
		strcat(fullname, "|");
		strcat(fullname, fp->filename.c_str());
	} else
	{
		strcpy(fullname, name);
	}

	//file opened ok. start loading.
    fceu::printf("Loading %s...\n\n", fullname);
	file.GetFileBase(fp->filename.c_str());
	ResetGameLoaded();
	//reset parameters so they're cleared just in case a format's loader doesn't know to do the clearing
	ines.MasterRomInfoParams = TMasterRomInfoParams();

	if (!AutosaveStatus)
		AutosaveStatus = (int*)fceu::dmalloc(sizeof(int) * AutosaveQty);
	for (AutosaveIndex = 0; AutosaveIndex < AutosaveQty; ++AutosaveIndex)
		AutosaveStatus[AutosaveIndex] = 0;

	CloseGame();
	GameInfo = new FCEUGI();
	memset(GameInfo, 0, sizeof(FCEUGI));

	GameInfo->filename = strdup(fp->filename.c_str());
	if (fp->archiveFilename != "")
		GameInfo->archiveFilename = strdup(fp->archiveFilename.c_str());
	GameInfo->archiveCount = fp->archiveCount;

	GameInfo->soundchan = 0;
	GameInfo->soundrate = 0;
	GameInfo->name = 0;
	GameInfo->type = GIT_CART;
	GameInfo->vidsys = GIV_USER;
	GameInfo->input[0] = GameInfo->input[1] = SI_UNSET;
	GameInfo->inputfc = SIFC_UNSET;
	GameInfo->cspecial = SIS_NONE;

	//try to load each different format
	if (ines.iNESLoad(fullname, fp, OverwriteVidMode))
		goto endlseq;
	if (NSFLoad(fullname, fp))
		goto endlseq;
	if (UNIFLoad(fullname, fp))
		goto endlseq;
	if (fds.FDSLoad(fullname, fp))
		goto endlseq;

	if (!silent)
		fceu::PrintError("An error occurred while loading the file.");
    fceu::fclose(fp);

	delete GameInfo;
	GameInfo = 0;

	return 0;

 endlseq:

    fceu::fclose(fp);

	if (OverwriteVidMode)
		ResetVidSys();

	if (GameInfo->type != GIT_NSF)
	{
		if (FSettings.GameGenie)
		{
			if (cart.OpenGenie())
			{
				SetGameGenie(false);
			}
		}
	}
	PowerNES();

	if (GameInfo->type != GIT_NSF)
		FCEU_LoadGamePalette();

	FCEU_ResetPalette();
	FCEU_ResetMessages();   // Save state, status messages, etc.

	if (!lastpal && PAL) {
        fceu::DispMessage("PAL mode set", 0);
        fceu::printf("PAL mode set");
	} else if (!lastdendy && dendy) {
		// this won't happen, since we don't autodetect dendy, but maybe someday we will?
        fceu::DispMessage("Dendy mode set", 0);
        fceu::printf("Dendy mode set");
	} else if ((lastpal || lastdendy) && !(PAL || dendy)) {
        fceu::DispMessage("NTSC mode set", 0);
        fceu::printf("NTSC mode set");
	}

	if (GameInfo->type != GIT_NSF)
		cheat.LoadGameCheats(0);

	if (AutoResumePlay)
	{
		// load "-resume" savestate
		if (FCEUSS_Load(file.MakeFName(FCEUMKF_RESUMESTATE, 0, 0).c_str(), false))
			fceu::DispMessage("Old play session resumed.", 0);
	}

	ResetScreenshotsCounter();

	return GameInfo;
}

FCEUGI* FCEU::LoadGame(const char *name, int OverwriteVidMode, bool silent) {
	return LoadGameVirtual(name, OverwriteVidMode, silent);
}

//Return: Flag that indicates whether the function was succesful or not.
bool FCEU::Initialize() {
	srand(time(0));

	if (!FCEU_InitVirtualVideo()) {
		return false;
	}

	AllocBuffers();

	// Initialize some parts of the settings structure
	//mbg 5/7/08 - I changed the ntsc settings to match pal.
	//this is more for precision emulation, instead of entertainment, which is what fceux is all about nowadays
	memset(&FSettings, 0, sizeof(FSettings));
	//FSettings.UsrFirstSLine[0]=8;
	FSettings.UsrFirstSLine[0] = 0;
	FSettings.UsrFirstSLine[1] = 0;
	//FSettings.UsrLastSLine[0]=231;
	FSettings.UsrLastSLine[0] = 239;
	FSettings.UsrLastSLine[1] = 239;
	FSettings.SoundVolume = 150;      //0-150 scale
	FSettings.TriangleVolume = 256;   //0-256 scale (256 is max volume)
	FSettings.Square1Volume = 256;    //0-256 scale (256 is max volume)
	FSettings.Square2Volume = 256;    //0-256 scale (256 is max volume)
	FSettings.NoiseVolume = 256;      //0-256 scale (256 is max volume)
	FSettings.PCMVolume = 256;        //0-256 scale (256 is max volume)

	ppu.Init();

	x6502.Init();

	return true;
}

void FCEU::Kill(void) {
	FCEU_KillVirtualVideo();
	cart.KillGenie();
	FreeBuffers();
}

void FCEU::SetAutoFirePattern(int onframes, int offframes) {
	int i;
	for (i = 0; i < onframes && i < 8; i++) {
		AutoFirePattern[i] = 1;
	}
	for (; i < 8; i++) {
		AutoFirePattern[i] = 0;
	}
	if (onframes + offframes < 2) {
		AutoFirePatternLength = 2;
	} else if (onframes + offframes > 8) {
		AutoFirePatternLength = 8;
	} else {
		AutoFirePatternLength = onframes + offframes;
	}
	AFon = onframes; AFoff = offframes;
}

void FCEU::SetAutoFireOffset(int offset) {
	if (offset < 0 || offset > 8) return;
	AutoFireOffset = offset;
}

void FCEU::AutoFire(void) {
	if (justLagged == false)
		counter = (counter + 1) % (8 * 7 * 5 * 3);
	//If recording a movie, use the frame # for the autofire so the offset
	//doesn't get screwed up when loading.
	if (movie.Mode(MOVIEMODE_RECORD | MOVIEMODE_PLAY)) {
		rapidAlternator = AutoFirePattern[(AutoFireOffset + movie.GetFrame()) % AutoFirePatternLength]; //adelikat: TODO: Think through this, MOVIEMODE_FINISHED should not use movie data for auto-fire?
	} else {
		rapidAlternator = AutoFirePattern[(AutoFireOffset + counter) % AutoFirePatternLength];
	}
}

///Emulates a single frame.
///Skip may be passed in, if FRAMESKIP is #defined, to cause this to emulate more than one frame
void FCEU::Emulate(uint8 **pXBuf, int32 **SoundBuf, int32 *SoundBufSize, int skip) {
	//skip initiates frame skip if 1, or frame skip and sound skip if 2
	int r, ssize;

	JustFrameAdvanced = false;

	if (frameAdvanceRequested)
	{
		if (frameAdvance_Delay_count == 0 || frameAdvance_Delay_count >= frameAdvance_Delay)
			EmulationPaused_ = EMULATIONPAUSED_FA;
		if (frameAdvance_Delay_count < frameAdvance_Delay)
			frameAdvance_Delay_count++;
	}

	if (EmulationPaused_ & EMULATIONPAUSED_FA)
	{
		// the user is holding Frame Advance key
		// clear paused flag temporarily
		EmulationPaused_ &= ~EMULATIONPAUSED_PAUSED;
	} else
	{
		if (EmulationPaused_ & EMULATIONPAUSED_PAUSED)
		{
			// emulator is paused
			memcpy(XBuf, XBackBuf, 256*256);
			FCEU_PutImage();
			*pXBuf = XBuf;
			*SoundBuf = WaveFinal;
			*SoundBufSize = 0;
			return;
		}
	}

	AutoFire();
	UpdateAutosave();

	input.Update();
	movie.SetLagFlag(1);

	if (cart.GenieStage() != 1) cheat.ApplyPeriodicCheats();
	r = ppu.Loop(skip);

	if (skip != 2) ssize = FlushEmulateSound();  //If skip = 2 we are skipping sound processing

	timestampbase += x6502.timestamp();
	x6502.setTimestamp(0);
	x6502.setSoundtimestamp(0);

	*pXBuf = skip ? 0 : XBuf;
	if (skip == 2) { //If skip = 2, then bypass sound
		*SoundBuf = 0;
		*SoundBufSize = 0;
	} else {
		*SoundBuf = WaveFinal;
		*SoundBufSize = ssize;
	}

	if ((EmulationPaused_ & EMULATIONPAUSED_FA) && (!movie.GetFrameAdvanceLagSkip() || !movie.GetLagged()))
	//Lots of conditions here.  EmulationPaused_ & EMULATIONPAUSED_FA must be true.  In addition frameAdvanceLagSkip or lagFlag must be false
	// When Frame Advance is held, emulator is automatically paused after emulating one frame (or several lag frames)
	{
		EmulationPaused_ = EMULATIONPAUSED_PAUSED;		   // restore EMULATIONPAUSED_PAUSED flag and clear EMULATIONPAUSED_FA flag
		JustFrameAdvanced = true;
	}

	if (movie.GetLagged()) {
		movie.IncreaseLagCount();
		justLagged = true;
	} else justLagged = false;

	if (movieSubtitles)
		movie.ProcessSubtitles();
}

void FCEU::ResetNES(void) {
	movie.AddCommand(FCEUNPCMD_RESET);
	if (!GameInfo) return;
	(*GameInterface)(GI_RESETM2);
	FCEUSND_Reset();
	ppu.Reset();
	x6502.Reset();

	// clear back baffer
	memset(XBackBuf, 0, 256 * 256);

    fceu::DispMessage("Reset", 0);
}

void FCEU::MemoryRand(uint8 *ptr, uint32 size) {
	int x = 0;
	while (size) {
		uint8 v = 0;
		switch (RAMInitOption)
		{
			default:
			case 0: v = (x & 4) ? 0xFF : 0x00; break;
			case 1: v = 0xFF; break;
			case 2: v = 0x00; break;
			case 3: v = uint8(rand()); break;

			// the default is this 8 byte pattern: 00 00 00 00 FF FF FF FF
			// it has been used in FCEUX since time immemorial

			// Some games to examine uninitialied RAM problems with:
			// * Cybernoid - music option starts turned off with default pattern
			// * Huang Di - debug mode is enabled with default pattern
			// * Minna no Taabou no Nakayoshi Daisakusen - fails to boot with some patterns
			// * F-15 City War - high score table
			// * 1942 - high score table
			// * Cheetahmen II - may start in different levels with different RAM startup
		}
		*ptr = v;
		x++;
		size--;
		ptr++;
	}
}

void FCEU::PowerNES(void) {
	movie.AddCommand(FCEUNPCMD_POWER);
	if (!GameInfo) return;

    cheat.CheatResetRAM();
    cheat.CheatAddRAM(2, 0, RAM);

	cart.GeniePower();

	MemoryRand(RAM, 0x800);

	ppu.Power();
	input.Initialize();
	FCEUSND_Power();

	//Have the external game hardware "powered" after the internal NES stuff.  Needed for the NSF code and VS System code.
	(*GameInterface)(GI_POWER);
	if (GameInfo->type == GIT_VSUNI)
		FCEU_VSUniPower();

	//if we are in a movie, then reset the saveram
	if (cart.GetDisableBatteryLoading())
		(*GameInterface)(GI_RESETSAVE);

	timestampbase = 0;
	x6502.Power();
    cheat.PowerCheats();
	movie.LagCounterReset();
	// clear back buffer
	memset(XBackBuf, 0, 256 * 256);


    fceu::DispMessage("Power on", 0);
}

void FCEU::ResetVidSys(void) {
	int w;

	if (GameInfo->vidsys == GIV_NTSC)
		w = 0;
	else if (GameInfo->vidsys == GIV_PAL) {
		w = 1;
		dendy = 0;
	} else
		w = FSettings.PAL;

	PAL = w ? 1 : 0;

	if (PAL)
		dendy = 0;

    int newppu = ppu.is_newppu();
	if (newppu)
		overclock_enabled = 0;

	normalscanlines = (dendy ? 290 : 240)+newppu; // use flag as number!
	totalscanlines = normalscanlines + (overclock_enabled ? postrenderscanlines : 0);
	ppu.SetVideoSystem(w || dendy);
	SetSoundVariables();
}

void FCEU::SetRenderedLines(int ntscf, int ntscl, int palf, int pall) {
	FSettings.UsrFirstSLine[0] = ntscf;
	FSettings.UsrLastSLine[0] = ntscl;
	FSettings.UsrFirstSLine[1] = palf;
	FSettings.UsrLastSLine[1] = pall;
	if (PAL || dendy) {
		FSettings.FirstSLine = FSettings.UsrFirstSLine[1];
		FSettings.LastSLine = FSettings.UsrLastSLine[1];
	} else {
		FSettings.FirstSLine = FSettings.UsrFirstSLine[0];
		FSettings.LastSLine = FSettings.UsrLastSLine[0];
	}
}

void FCEU::SetVidSystem(int a) {
	FSettings.PAL = a ? 1 : 0;
	if (GameInfo) {
		ResetVidSys();
		FCEU_ResetPalette();
	}
}

int FCEU::GetCurrentVidSystem(int *slstart, int *slend) {
	if (slstart)
		*slstart = FSettings.FirstSLine;
	if (slend)
		*slend = FSettings.LastSLine;
	return(PAL);
}

void FCEU::SetRegion(int region, int notify) {
	switch (region) {
		case 0: // NTSC
			normalscanlines = 240;
			pal_emulation = 0;
			dendy = 0;
			break;
		case 1: // PAL
			normalscanlines = 240;
			pal_emulation = 1;
			dendy = 0;
			break;
		case 2: // Dendy
			normalscanlines = 290;
			pal_emulation = 0;
			dendy = 1;
			break;
	}
	normalscanlines += ppu.is_newppu();
	totalscanlines = normalscanlines + (overclock_enabled ? postrenderscanlines : 0);
	SetVidSystem(pal_emulation);
}

int32 FCEU::GetDesiredFPS(void) {
	if (PAL || dendy)
		return(838977920);  // ~50.007
	else
		return(1008307711);  // ~60.1
}

void FCEU::UpdateAutosave(void) {
	if (!EnableAutosave || turbo)
		return;

	char * f;
	if (++AutosaveCounter >= AutosaveFrequency) {
		AutosaveCounter = 0;
		AutosaveIndex = (AutosaveIndex + 1) % AutosaveQty;
		f = strdup(file.MakeFName(FCEUMKF_AUTOSTATE, AutosaveIndex, 0).c_str());
		FCEUSS_Save(f, false);
		AutoSS = true;  //Flag that an auto-savestate was made
        fceu::free(f);
        f = NULL;
		AutosaveStatus[AutosaveIndex] = 1;
	}
}

void FCEU::RewindToLastAutosave(void) {
	if (!EnableAutosave || !AutoSS)
		return;

	if (AutosaveStatus[AutosaveIndex] == 1) {
		char * f;
		f = strdup(file.MakeFName(FCEUMKF_AUTOSTATE, AutosaveIndex, 0).c_str());
		FCEUSS_Load(f);
        fceu::free(f);
        f = NULL;

		//Set pointer to previous available slot
		if (AutosaveStatus[(AutosaveIndex + AutosaveQty - 1) % AutosaveQty] == 1) {
			AutosaveIndex = (AutosaveIndex + AutosaveQty - 1) % AutosaveQty;
		}

		//Reset time to next Auto-save
		AutosaveCounter = 0;
	}
}

bool FCEU::IsValidUI(EFCEUI ui) {
	switch (ui) {
	case FCEUI_OPENGAME:
	case FCEUI_CLOSEGAME:
		if (movie.Mode(MOVIEMODE_TASEDITOR)) return false;
		break;
	case FCEUI_RECORDMOVIE:
	case FCEUI_PLAYMOVIE:
	case FCEUI_QUICKSAVE:
	case FCEUI_QUICKLOAD:
	case FCEUI_SAVESTATE:
	case FCEUI_LOADSTATE:
	case FCEUI_NEXTSAVESTATE:
	case FCEUI_PREVIOUSSAVESTATE:
	case FCEUI_VIEWSLOTS:
		if (!GameInfo) return false;
		if (movie.Mode(MOVIEMODE_TASEDITOR)) return false;
		break;

	case FCEUI_STOPMOVIE:
		return(movie.Mode(MOVIEMODE_PLAY | MOVIEMODE_RECORD | MOVIEMODE_FINISHED));

	case FCEUI_PLAYFROMBEGINNING:
		return(movie.Mode(MOVIEMODE_PLAY | MOVIEMODE_RECORD | MOVIEMODE_TASEDITOR | MOVIEMODE_FINISHED));

	case FCEUI_STOPAVI:
		return false;

	case FCEUI_TASEDITOR:
		if (!GameInfo) return false;
		break;

	case FCEUI_RESET:
	case FCEUI_POWER:
	case FCEUI_EJECT_DISK:
	case FCEUI_SWITCH_DISK:
	case FCEUI_INSERT_COIN:
		if (!GameInfo) return false;
		if (movie.Mode(MOVIEMODE_RECORD)) return true;
		if (!movie.Mode(MOVIEMODE_INACTIVE)) return false;
		break;
	}
	return true;
}

} // namespace fceu
