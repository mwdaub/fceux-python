#include "input_obj.h"

#include "fceu_obj.h"

namespace fceu {

void Input::Update(void) {
	//tell all drivers to poll input and set up their logical states
	if(!fceu->movie.Mode(MOVIEMODE_PLAY))
	{
		for(int port=0;port<2;port++){
			joyports[port].driver->Update(port,joyports[port].ptr,joyports[port].attrib);
		}
	}

	if(fceu->GameInfo->type==GIT_VSUNI)
		if(coinon) coinon--;

	fceu->movie.AddInputState();

	//TODO - should this apply to the movie data? should this be displayed in the input hud?
	if(fceu->GameInfo->type==GIT_VSUNI){
		FCEU_VSUniSwap(&joy[0],&joy[1]);
	}
}

//initializes the input system to power-on state
void Input::Initialize(void) {
	memset(joy_readbit,0,sizeof(joy_readbit));
	memset(joy,0,sizeof(joy));
	LastStrobe = 0;

	if(fceu->GameInfo->type==GIT_VSUNI) {
		fceu->handler.SetReadHandler(0x4016,0x4016,&VSUNIRead0_);
		fceu->handler.SetReadHandler(0x4017,0x4017,&VSUNIRead1_);
	} else {
		fceu->handler.SetReadHandler(0x4016,0x4017,&JPRead_);
    }

	fceu->handler.SetWriteHandler(0x4016,0x4016,&B4016_);

	//force the port drivers to be setup
	SetDriver(0);
	SetDriver(1);
}

//binds JPorts[pad] to the driver specified in JPType[pad]
void Input::SetDriver(int port) {
	switch(joyports[port].type) {
	case SI_GAMEPAD:
		if(fceu->GameInfo->type==GIT_VSUNI){
			joyports[port].driver = &GPCVS;
		} else {
			joyports[port].driver= &GPC;
		}
		break;
	case SI_NONE:
		joyports[port].driver=&DummyJPort;
		break;
	}
}

//This function is a quick hack to get the NSF player to use emulated gamepad input.
uint8 Input::GetJoyJoy(void)
{
	return(joy[0]|joy[1]|joy[2]|joy[3]);
}

uint8 Input::VSUNIRead0(uint32 A) {
	fceu->movie.SetLagFlag(0);
	uint8 ret=0;

	ret|=(joyports[0].driver->Read(0))&1;

	ret|=(vsdip&3)<<3;
	if(coinon)
		ret|=0x4;
	return ret;
}

uint8 Input::VSUNIRead1(uint32 A) {
	fceu->movie.SetLagFlag(0);
	uint8 ret=0;

	ret|=(joyports[1].driver->Read(1))&1;
	ret|=vsdip&0xFC;
	return ret;
}

uint8 Input::JPRead(uint32 A) {
	fceu->movie.SetLagFlag(0);
	uint8 ret=0;

	ret|=joyports[A&1].driver->Read(A&1);

	ret|=fceu->x6502.DB()&0xC0;

	return(ret);
}

void Input::B4016(uint32 A, uint8 V) {
	for(int i=0;i<2;i++)
		joyports[i].driver->Write(V&1);

	if((LastStrobe&1) && (!(V&1)))
	{
		//old comment:
		//This strobe code is just for convenience.  If it were
		//with the code in input / *.c, it would more accurately represent
		//what's really going on.  But who wants accuracy? ;)
		//Seriously, though, this shouldn't be a problem.
		//new comment:

		//mbg 6/7/08 - I guess he means that the input drivers could track the strobing themselves
		//I dont see why it is unreasonable here.
		for(int i=0;i<2;i++)
			joyports[i].driver->Strobe(i);
	}
	LastStrobe=V&0x1;
	RawReg4016 = V;
}

//calls from the ppu;
//calls the SLHook for any driver that needs it
void Input::ScanlineHook(uint8 *bg, uint8 *spr, uint32 linets, int final)
{
	for(int port=0;port<2;port++)
		joyports[port].driver->SLHook(port,bg,spr,linets,final);
}

//basic joystick port driver
uint8 Input::ReadGP(int w) {
	uint8 ret;

	if(joy_readbit[w]>=8)
		ret = ((joy[2+w]>>(joy_readbit[w]&7))&1);
	else
		ret = ((joy[w]>>(joy_readbit[w]))&1);
	if(joy_readbit[w]>=16) ret=0;
	if(!FSAttached)
	{
		if(joy_readbit[w]>=8) ret|=1;
	}
	else
	{
		if(joy_readbit[w]==19-w) ret|=1;
	}
	if(fceu->ppu.get_fceuindbg())
		joy_readbit[w]++;
	return ret;
}

uint8 Input::ReadGPVS(int w) {
	uint8 ret=0;

	if(joy_readbit[w]>=8)
		ret=1;
	else
	{
		ret = ((joy[w]>>(joy_readbit[w]))&1);
		if(!fceu->ppu.get_fceuindbg())
			joy_readbit[w]++;
	}
	return ret;
}

void Input::StrobeGP(int w) {
	joy_readbit[w]=0;
}

void Input::LoadGP(int w, MovieRecord* mr) {
	if(w==0)
	{
		joy[0] = mr->joysticks[0];
		if(FSAttached) joy[2] = mr->joysticks[2];
	}
	else
	{
		joy[1] = mr->joysticks[1];
		if(FSAttached) joy[3] = mr->joysticks[3];
	}
}

void Input::UpdateGP(int w, void *data, int arg) {
	if(w==0)	//adelikat, 3/14/09: Changing the joypads to inclusive OR the user's joypad + the Lua joypad, this way lua only takes over the buttons it explicity says to
	{			//FatRatKnight: Assume lua is always good. If it's doing nothing in particular using my logic, it'll pass-through the values anyway.
		joy[0] = *(uint32 *)joyports[0].ptr;;
		joy[2] = *(uint32 *)joyports[0].ptr >> 16;
	}
	else
	{
		joy[1] = *(uint32 *)joyports[1].ptr >> 8;
		joy[3] = *(uint32 *)joyports[1].ptr >> 24;
	}
}

void Input::LogGP(int w, MovieRecord* mr) {
	if(w==0)
	{
		mr->joysticks[0] = joy[0];
		mr->joysticks[2] = joy[2];
	}
	else
	{
		mr->joysticks[1] = joy[1];
		mr->joysticks[3] = joy[3];
	}
}

} // namespace fceu
