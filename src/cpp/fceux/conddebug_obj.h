#ifndef CONDDEBUG_H
#define CONDDEBUG_H

#define TYPE_NO 0
#define TYPE_REG 1
#define TYPE_FLAG 2
#define TYPE_NUM 3
#define TYPE_ADDR 4
#define TYPE_PC_BANK 5
#define TYPE_DATA_BANK 6
#define TYPE_VALUE_READ 7
#define TYPE_VALUE_WRITE 8

#define OP_NO 0
#define OP_EQ 1
#define OP_NE 2
#define OP_GE 3
#define OP_LE 4
#define OP_G 5
#define OP_L 6
#define OP_PLUS 7
#define OP_MINUS 8
#define OP_MULT 9
#define OP_DIV 10
#define OP_OR 11
#define OP_AND 12

namespace fceu {

uint16 debugLastAddress;
uint8 debugLastOpcode;

//mbg merge 7/18/06 turned into sane c++
struct Condition
{
	Condition* lhs;
	Condition* rhs;

	unsigned int type1;
	unsigned int value1;

	unsigned int op;

	unsigned int type2;
	unsigned int value2;
};

void freeTree(Condition* c);
Condition* generateCondition(const char* str);

} // namespace fceu

#endif // define CONDDEBUG_H
