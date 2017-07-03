#include <stdio.h>

#include "fceuxmodule.h"

// Python error for when a file is not a valid ROM format.
static PyObject *InvalidRomError;

// Game input.
// "A" = 1, "B" = 2, "SELECT" = 4, "START" = 8, "UP" = 16, "DOWN" = 32, "LEFT" = 64, "RIGHT" = 128
// Combinations of buttons are indicated by taking their sums.
// To specify input for player 2, take the identical player 1 input and bit shift left by 8.
uint32 input;

// Load a rom and, optionally, start recording an fceux movie. Since the emulation is
// deterministic, the movie only needs to save the controller input at each frame. The movie can be
// replayed in the fceux emulator by running:
// fceux --playmov <movie_file_name> <rom_file_name>
//
// Arguments: romFileName, movieFileName
static PyObject * fceux_load_rom(PyObject *self, PyObject *args) {
  char * romFileName;
  char * movieFileName = "";

  // parse arguments
  if (!PyArg_ParseTuple(args, "s|s", &romFileName, &movieFileName)) {
    return NULL;
  }

  if (!FCEUI_LoadGame(romFileName, 1)) {
    char *msg;
    asprintf(&msg,"File %s is not a valid ROM or does not exist.", romFileName);
    PyErr_SetString(InvalidRomError, msg);
    return NULL;
  }
  FCEUI_SetInput(0, SI_GAMEPAD, &input, 0);
  if (!(movieFileName[0] == '\0')) {
    FCEUI_SaveMovie(movieFileName, MOVIE_FLAG_FROM_POWERON, L"");
  }

  Py_INCREF(Py_None);
  return Py_None;
}

// Close a rom and, if a movie is recording, stop recording the movie.
static PyObject * fceux_close_rom(PyObject *self, PyObject *args) {
  // parse arguments
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  FCEUI_CloseGame();

  Py_INCREF(Py_None);
  return Py_None;
}

// Dimensions of the game image.
static long int dims[] = {240, 256, 3};

// Emulate a single frame, returning the pixels of the game screen.
// TODO: return sound data as well.
//
// Arguments: integer representing controller input for the frame. See comment above the input
// variable for details.
// Returns: NumPy array with shape dims and type uint8, containing the pixel data.
static PyObject * fceux_emulate_frame(PyObject *self, PyObject *args) {
  // parse arguments
  if (!PyArg_ParseTuple(args, "I", &input)) {
    return NULL;
  }

  uint8 *gfx;
  int32 *sound;
  int32 ssize;
  FCEUI_Emulate(&gfx, &sound, &ssize, 0);

  PyObject *array = PyArray_SimpleNew(3, dims, NPY_UINT8);
  uint8 *data = (unsigned char *) PyArray_DATA(array);
  for (size_t i = 0; i < 240*256; i++) {
    uint8 idx = gfx[i] - 128;
    data[3*i] = palo[idx].r;
    data[3*i+1] = palo[idx].g;
    data[3*i+2] = palo[idx].b;
  }
  return array;
}

// Read the value at the given memory address.
//
// Arguments: uint16 representing the memory address.
// Returns: uint8 representing the memory value.
static PyObject * fceux_read_memory(PyObject *self, PyObject *args) {
  uint16 memAddress;

  // parse arguments
  if (!PyArg_ParseTuple(args, "H", &memAddress)) {
    return NULL;
  }

  // read memory
  uint8 memValue = GetMem(memAddress);
  return Py_BuildValue("I", memValue);
}

// List of fceux module methods.
static PyMethodDef FceuxMethods[] = {
  { "load_rom", fceux_load_rom, METH_VARARGS, "Load NES rom" },
  { "close_rom", fceux_close_rom, METH_VARARGS, "Close loaded NES rom" },
  { "emulate_frame", fceux_emulate_frame, METH_VARARGS, "Emulate a single frame" },
  { "read_memory", fceux_read_memory, METH_VARARGS, "Read the value at the given memory address" },
  { NULL, NULL, 0, NULL }
};

// Initialize the fceux module.
PyMODINIT_FUNC initfceux(void)
{
  PyObject *m;
  m = Py_InitModule("fceux", FceuxMethods);
  if (m == NULL) {
    return;
  }

  InvalidRomError = PyErr_NewException("fceux.invalid_rom", NULL, NULL);
  Py_INCREF(InvalidRomError);
  PyModule_AddObject(m, "error", InvalidRomError);

  import_array();
  FCEUI_Initialize();
}
