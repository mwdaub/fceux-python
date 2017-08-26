#include "memory_obj.h"

namespace FCEU {

///allocates the specified number of bytes. returns null if this fails
void* malloc(uint32 size) {
  void *ret;
  ret=malloc(size);
  if(!ret) {
    FCEU::PrintError("Error allocating memory!");
    return(0);
  }
  //mbg 6/17/08 - sometimes this memory is used as RAM or somesuch without clearing first.
  //this yields different behavior in debug and release modes.
  //specifically, saveram wasnt getting cleared so the games thought their savefiles were initialized
  //so we are going to clear it here.
  memset(ret,0,size);
  return ret;
}

///allocates the specified number of bytes. exits process if this fails
void* gmalloc(uint32 size) {
  void *ret;
  ret=malloc(size);
  if(!ret) {
    FCEU::PrintError("Error allocating memory!  Doing a hard exit.");
    exit(1);
  }
  //mbg 6/17/08 - sometimes this memory is used as RAM or somesuch without clearing first.
  //this yields different behavior in debug and release modes.
  //specifically, saveram wasnt getting cleared so the games thought their savefiles were initialized
  //so we are going to clear it here.
  memset(ret,0,size);
  return ret;
}

///frees memory allocated with FCEU_gmalloc
void gfree(void *ptr) {
  free(ptr);
}

// frees memory allocated with FCEU_malloc
// Might do something with this and FCEU_malloc later...
void free(void *ptr) {
  free(ptr);
}

void* dmalloc(uint32 size) {
  return malloc(size);
}

void dfree(void *ptr) {
  free(ptr);
}

} // namespace FCEU
