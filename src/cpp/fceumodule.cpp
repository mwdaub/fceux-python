#include <stdio.h>

#include "fceumodule.h"

// Python error for when a file is not a valid ROM format.
static PyObject *InvalidRomError;

// Python error for when the emulator state is in an illegal state during a function call.
static PyObject *IllegalStateError;

// Game input.
// "A" = 1, "B" = 2, "SELECT" = 4, "START" = 8, "UP" = 16, "DOWN" = 32, "LEFT" = 64, "RIGHT" = 128
// Combinations of buttons are indicated by taking their sums.
// To specify input for player 2, take the identical player 1 input and bit shift left by 8.
uint32 input;

static bool gameLoaded;

// Load a rom and, optionally, start recording an fceu movie. Since the emulation is
// deterministic, the movie only needs to save the controller input at each frame. The movie can be
// replayed in the fceu emulator by running:
// fceu --playmov <movie_file_name> <rom_file_name>
//
// Arguments: romFileName, movieFileName
static PyObject * fceu_load_game(PyObject *self, PyObject *args) {
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

  gameLoaded = true;

  Py_INCREF(Py_None);
  return Py_None;
}

static void set_game_error_msg() {
  char *msg = "No game is currently loaded.";
  PyErr_SetString(IllegalStateError, msg);
}

static PyObject * fceu_load_state(PyObject *self, PyObject *args) {
  char * stateFileName;

  // parse arguments
  if (!PyArg_ParseTuple(args, "s", &stateFileName)) {
    return NULL;
  }

  if (!gameLoaded) {
    set_game_error_msg();
    return NULL;
  }

  FCEUI_LoadState(stateFileName, false);

  Py_INCREF(Py_None);
  return Py_None;
}

// Close a game and, if a movie is recording, stop recording the movie.
static PyObject * fceu_close_game(PyObject *self, PyObject *args) {
  // parse arguments
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  FCEUI_CloseGame();

  gameLoaded = false;

  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject * fceu_save_state(PyObject *self, PyObject *args) {
  char * stateFileName;

  // parse arguments
  if (!PyArg_ParseTuple(args, "s", &stateFileName)) {
    return NULL;
  }

  if (!gameLoaded) {
    set_game_error_msg();
    return NULL;
  }

  FCEUI_SaveState(stateFileName, false);

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
static PyObject * fceu_emulate_frame(PyObject *self, PyObject *args) {
  uint8 frames = 1;

  // parse arguments
  if (!PyArg_ParseTuple(args, "I|B", &input, &frames)) {
    return NULL;
  }

  if (!gameLoaded) {
    set_game_error_msg();
    return NULL;
  }

  uint8 *gfx;
  int32 *sound;
  int32 ssize;
  for (uint8 i = 0; i < frames; i++) {
    FCEUI_Emulate(&gfx, &sound, &ssize, 0);
  }

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
// Key SMB values: 0x0756 size (0 = small, 1 = big, 2 = fire), 0x075A lives, 0x075C stage number, 0x075E coins, 0x075F world number
// 0x07F8 clock hundreds, 0x07F9 clock tens, 0x07FA clock ones
// 0x07D8 score hundred thousands, 0x07D9 score ten thousands, 0x07DA score thousands, 0x07DB score hundreds, 0x07DC score tens, 0x07DD score ones
// 0x07DE-0x07E3 also seems to contain the score.
//
// Arguments: uint16 representing the memory address.
// Returns: uint8 representing the memory value.
static PyObject * fceu_read_memory(PyObject *self, PyObject *args) {
  uint16 memAddress;

  // parse arguments
  if (!PyArg_ParseTuple(args, "H", &memAddress)) {
    return NULL;
  }

  if (!gameLoaded) {
    set_game_error_msg();
    return NULL;
  }

  // read memory
  uint8 memValue = GetMem(memAddress);
  return Py_BuildValue("B", memValue);
}

static bool search_initialized;
static int16 memoryValues[0x0800];

static PyObject * fceu_init_memory_search(PyObject *self, PyObject *args) {
  // parse arguments
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }

  if (!gameLoaded) {
    set_game_error_msg();
    return NULL;
  }

  for (int i = 0; i < 0x0800; i++) {
    memoryValues[i] = GetMem((uint16) i);
  }
  search_initialized = true;

  Py_INCREF(Py_None);
  return Py_None;
}

static void set_memory_error_msg() {
  char *msg = "Memory search has not been initialized; call init_memory_search first.";
  PyErr_SetString(IllegalStateError, msg);
}

static PyObject * fceu_match_unchanged(PyObject *self, PyObject *args) {
  // parse arguments
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }

  if (!gameLoaded) {
    set_game_error_msg();
    return NULL;
  }

  if (!search_initialized) {
    set_memory_error_msg();
    return NULL;
  }

  uint16 count = 0;
  for (int i = 0; i < 0x0800; i++) {
    int16 previousMemValue = memoryValues[i];
    if (previousMemValue < 0) {
      continue;
    }
    uint8 currentMemValue = GetMem((uint16) i);
    if (((uint8) previousMemValue) != currentMemValue) {
      memoryValues[i] = -1;
    } else {
      count++;
    }
  }

  return Py_BuildValue("H", count);
}

static PyObject * fceu_match_changed(PyObject *self, PyObject *args) {
  // parse arguments
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }

  if (!gameLoaded) {
    set_game_error_msg();
    return NULL;
  }

  if (!search_initialized) {
    set_memory_error_msg();
    return NULL;
  }

  uint16 count = 0;
  for (int i = 0; i < 0x0800; i++) {
    int16 previousMemValue = memoryValues[i];
    if (previousMemValue < 0) {
      continue;
    }
    uint8 currentMemValue = GetMem((uint16) i);
    if (((uint8) previousMemValue) == currentMemValue) {
      memoryValues[i] = -1;
    } else {
      memoryValues[i] = currentMemValue;
      count++;
    }
  }

  return Py_BuildValue("H", count);
}

static PyObject * fceu_match_equals(PyObject *self, PyObject *args) {
  uint8 filterValue;
  // parse arguments
  if (!PyArg_ParseTuple(args, "B", &filterValue)) {
    return NULL;
  }

  if (!gameLoaded) {
    set_game_error_msg();
    return NULL;
  }

  if (!search_initialized) {
    set_memory_error_msg();
    return NULL;
  }

  uint16 count = 0;
  for (int i = 0; i < 0x0800; i++) {
    int16 previousMemValue = memoryValues[i];
    if (previousMemValue < 0) {
      continue;
    }
    uint8 currentMemValue = GetMem((uint16) i);
    if (currentMemValue != filterValue) {
      memoryValues[i] = -1;
    } else {
      memoryValues[i] = currentMemValue;
      count++;
    }
  }

  return Py_BuildValue("H", count);
}

static PyObject * fceu_get_matches(PyObject *self, PyObject *args) {
  // parse arguments
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }

  if (!gameLoaded) {
    set_game_error_msg();
    return NULL;
  }

  if (!search_initialized) {
    set_memory_error_msg();
    return NULL;
  }

  PyObject *matches = PyDict_New();
  for (int i = 0; i < 0x0800; i++) {
    int16 memValue = memoryValues[i];
    if (memValue < 0) {
      continue;
    }
    PyObject *pyKey = Py_BuildValue("H", (uint16) i);
    PyObject *pyValue = Py_BuildValue("B", GetMem((uint16) i));
    PyDict_SetItem(matches, pyKey, pyValue);
  }

  return matches;
}

// List of fceu module methods.
static PyMethodDef FceuxMethods[] = {
  { "load_game", fceu_load_game, METH_VARARGS, "Load NES game." },
  { "load_state", fceu_load_state, METH_VARARGS, "Load NES savestate." },
  { "close_game", fceu_close_game, METH_VARARGS, "Close loaded NES game." },
  { "save_state", fceu_save_state, METH_VARARGS, "Save NES savestate." },
  { "emulate_frame", fceu_emulate_frame, METH_VARARGS, "Emulate a single frame." },
  { "read_memory", fceu_read_memory, METH_VARARGS, "Read the value at the given memory address." },
  { "init_memory_search", fceu_init_memory_search, METH_VARARGS, "Initialize a search of memory values." },
  { "match_unchanged", fceu_match_unchanged, METH_VARARGS, "Matches memory values unchanged since the last search operation." },
  { "match_changed", fceu_match_changed, METH_VARARGS, "Matches memory values changed since the last search operation." },
  { "match_equals", fceu_match_equals, METH_VARARGS, "Matches memory values equal to a target value." },
  { "get_matches", fceu_get_matches, METH_VARARGS, "Get the results of the memory search." },
  { NULL, NULL, 0, NULL }
};

// Initialize the fceu module.
PyMODINIT_FUNC initfceu(void)
{
  PyObject *m;
  m = Py_InitModule("fceu", FceuxMethods);
  if (m == NULL) {
    return;
  }

  InvalidRomError = PyErr_NewException("fceu.invalid_rom", NULL, NULL);
  Py_INCREF(InvalidRomError);
  PyModule_AddObject(m, "error", InvalidRomError);

  IllegalStateError = PyErr_NewException("fceu.illegal_state", NULL, NULL);
  Py_INCREF(IllegalStateError);
  PyModule_AddObject(m, "error", IllegalStateError);

  import_array();
  FCEUI_Initialize();
}
