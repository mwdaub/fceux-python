#ifndef _INPUT_H_
#define _INPUT_H_

#include <iostream>

#include "types_obj.h"
#include "git_obj.h"

#include "utils/memory_obj.h"

#include "handler_obj.h"
#include "movie_obj.h"
#include "vsuni_obj.h"
#include "x6502_obj.h"

namespace fceu {

unsigned int lagCounter;
char lagFlag;

void LagCounterReset();

typedef std::function<uint8(int)> inputread;
typedef std::function<void(uint8)> inputwrite;
typedef std::function<void(int)> inputstrobe;
typedef std::function<void(int,void*,int)> inputupdate;
typedef std::function<void(int,uint8*,uint8*,uint32,int)> inputhook;
typedef std::function<void(int,uint8*,int)> inputdraw;
typedef std::function<void(int,MovieRecord*)> inputlog;
typedef std::function<void(int,MovieRecord*)> inputload;

//The interface for standard joystick port device drivers
struct INPUTC
{
	//these methods call the function pointers (or not, if they are null)
	uint8 Read(int w) { if(Read_) return (*Read_)(w); else return 0; }
	void Write(uint8 w) { if(Write_) (*Write_)(w); }
	void Strobe(int w) { if(Strobe_) (*Strobe_)(w); }
	void Update(int w, void *data, int arg) { if(Update_) (*Update_)(w,data,arg); }
	void SLHook(int w, uint8 *bg, uint8 *spr, uint32 linets, int final) { if(SLHook_) (*SLHook_)(w,bg,spr,linets,final); }
	void Draw(int w, uint8 *buf, int arg) { if(Draw_) (*Draw_)(w,buf,arg); }
	void Log(int w, MovieRecord* mr) { if(Log_) (*Log_)(w,mr); }
	void Load(int w, MovieRecord* mr) { if(Load_) (*Load_)(w,mr); }

	inputread* Read_;
    inputwrite* Write_;
    inputstrobe* Strobe_;
	//update will be called if input is coming from the user. refresh your logical state from user input devices
    inputupdate* Update_;
    inputhook* SLHook_;
    inputdraw* Draw_;

	//log is called when you need to put your logical state into a movie record for recording
    inputlog* Log_;
	//load will be called if input is coming from a movie. refresh your logical state from a movie record
    inputload* Load_;
};

struct JOYPORT {
	int w;
	int attrib;
	ESI type;
	void* ptr;
	INPUTC* driver;

	void log(MovieRecord* mr) { driver->Log(w,mr); }
	void load(MovieRecord* mr) { driver->Load(w,mr); }
};

class Input {
  public:
    Input(Handler* handler) : handler_(handler) {
      joyports[1].w = 1;
    }

    //called from PPU on scanline events.
    void ScanlineHook(uint8 *bg, uint8 *spr, uint32 linets, int final);

    void Initialize(void);
    void Update(void);

    bool GetFourscore() { return FSAttached; };
    void SetFourscore(bool attachFourscore) { FSAttached = attachFourscore; };

  private:
    // Members.
    Handler* handler_;
    X6502* x6502_;

    FCEUGI** GameInfo;

    int fceuindbg;

    JOYPORT joyports[2];

    INPUTC GPC={&ReadGP_,0,&StrobeGP_,&UpdateGP_,0,0,&LogGP_,&LoadGP_};
    INPUTC GPCVS={&ReadGPVS_,0,&StrobeGP_,&UpdateGP_,0,0,&LogGP_,&LoadGP_};
    INPUTC DummyJPort={0};

    uint8 joy_readbit[2];
    uint8 joy[4];
    uint16 snesjoy[4];
    uint8 LastStrobe;
    uint8 RawReg4016 = 0; // Joystick strobe (W)

    //set to true if the fourscore is attached
    bool FSAttached = false;

    // Methods.
    void SetDriver(int port);

    uint8 ReadGP(int w);
    uint8 ReadGPVS(int w);
    void StrobeGP(int w);
    void LoadGP(int w, MovieRecord* mr);
    void UpdateGP(int w, void *data, int arg);
    void LogGP(int w, MovieRecord* mr);

    inputread ReadGP_ = [this](int w) { return this->ReadGP(w); };
    inputread ReadGPVS_ = [this](int w) { return this->ReadGPVS(w); };
    inputstrobe StrobeGP_ = [this](int w) { return this->StrobeGP(w); };
    inputload LoadGP_ = [this](int w, MovieRecord* mr) { return this->LoadGP(w, mr); };
    inputupdate UpdateGP_ = [this](int w, void* data, int arg) { return this->UpdateGP(w, data, arg); };
    inputlog LogGP_ = [this](int w, MovieRecord* mr) { return this->LogGP(w, mr); };

    uint8 VSUNIRead0(uint32 A);
    uint8 VSUNIRead1(uint32 A);
    uint8 JPRead(uint32 A);
    void B4016(uint32 A, uint8 V);

    readfunc VSUNIRead0_ = [this](uint32 A) { return this->VSUNIRead0(A); };
    readfunc VSUNIRead1_ = [this](uint32 A) { return this->VSUNIRead1(A); };
    readfunc JPRead_ = [this](uint32 A) { return this->JPRead(A); };
    writefunc B4016_ = [this](uint32 A, uint8 V) { return this->B4016(A, V); };
};

} // namespace fceu

#endif // _INPUT_H_
