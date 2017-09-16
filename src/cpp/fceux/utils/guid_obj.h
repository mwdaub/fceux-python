#ifndef _guid_h_
#define _guid_h_

#include <string>
#include <stdlib.h>

#include "../types_obj.h"
#include "valuearray_obj.h"

namespace fceu {

struct Guid : public ValueArray<uint8,16>
{
	void newGuid();
	std::string toString();
	static Guid fromString(std::string str);
	static uint8 hexToByte(char** ptrptr);
	void scan(std::string& str);
};

} // namespace fceu

#endif // define _guid_h_
