#ifndef _MEMORY_H_
#define _MEMORY_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "../types_obj.h"

namespace fceu {

#define FCEU_dwmemset(d,c,n) {int _x; for(_x=n-4;_x>=0;_x-=4) *(uint32 *)&(d)[_x]=c;}

void* malloc(uint32 size);
void* gmalloc(uint32 size);
void gfree(void *ptr);
void free(void *ptr);
void memmove(void *d, void *s, uint32 l);

// wrapper for debugging when its needed, otherwise act like
// normal malloc/free
void* dmalloc(uint32 size);
void dfree(void *ptr);

} // namespace fceu

#endif // define _MEMORY_H_
