#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>

#include "fceux/git.h"
#include "fceux/movie.h"
#include "fceux/palette.h"

// Forward declaration of required FCEU functions.
bool FCEUI_Initialize();
void FCEUI_SetInput(int port, ESI type, void *ptr, int attrib);
FCEUGI *FCEUI_LoadGame(const char *name, int OverwriteVidMode, bool silent = false);
void FCEUI_SaveState(const char *fname, bool display_message=true);
void FCEUI_LoadState(const char *fname, bool display_message=true);
void FCEUI_CloseGame();
void ResetNES();
void FCEUI_SaveMovie(const char *filename, EMOVIE_FLAG flags, std::wstring author);
void FCEUI_Emulate(uint8 **pXBuf, int32 **SoundBuf, int32 *SoundBufSize, int skip);
uint8 GetMem(uint16 addr);

// Python error for when a file is not a valid ROM format.
static PyObject *InvalidRomError;

// Python error for when the emulator state is in an illegal state during a function call.
static PyObject *IllegalStateError;

// FCEUObject.
typedef struct {
  PyObject_HEAD
} FCEUObject;

// FCEUObject dealloc method.
static void FCEUObject_dealloc(FCEUObject* self);

// FCEUObject init method.
static int FCEUObject_init(FCEUObject *self, PyObject *args, PyObject *kwds);

// Game input.
// "A" = 1, "B" = 2, "SELECT" = 4, "START" = 8, "UP" = 16, "DOWN" = 32, "LEFT" = 64, "RIGHT" = 128
// Combinations of buttons are indicated by taking their sums.
// To specify input for player 2, take the identical player 1 input and bit shift left by 8.
uint32 input;

static bool gameLoaded;

// Dimensions of the game image.
static long int dims[] = {240, 256, 3};

static bool search_initialized;
static int16 memoryValues[0x0800];

static void FCEUObject_dealloc(FCEUObject* self) {
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static int FCEUObject_init(FCEUObject *self, PyObject *args, PyObject *kwds) {
  FCEUI_Initialize();
  return 0;
}

static PyObject * FCEUObject_load_game(FCEUObject* self, PyObject* args) {
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

static PyObject * FCEUObject_load_state(FCEUObject *self, PyObject *args) {
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
static PyObject * FCEUObject_close_game(FCEUObject* self, PyObject *args) {
  // parse arguments
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  FCEUI_CloseGame();

  gameLoaded = false;

  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject* FCEUObject_reset(FCEUObject* self, PyObject* args) {
  // parse arguments
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  ResetNES();

  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject * FCEUObject_save_state(FCEUObject* self, PyObject *args) {
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

// Emulate a single frame, returning the pixels of the game screen.
// TODO: return sound data as well.
//
// Arguments: integer representing controller input for the frame. See comment above the input
// variable for details.
// Returns: NumPy array with shape dims and type uint8, containing the pixel data.
static PyObject * FCEUObject_emulate_frame(FCEUObject* self, PyObject *args) {
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
static PyObject * FCEUObject_read_memory(FCEUObject* self, PyObject *args) {
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

static PyObject * FCEUObject_init_memory_search(FCEUObject* self, PyObject *args) {
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

static PyObject * FCEUObject_match_unchanged(FCEUObject* self, PyObject *args) {
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

static PyObject * FCEUObject_match_changed(FCEUObject* self, PyObject *args) {
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

static PyObject * FCEUObject_match_equals(FCEUObject* self, PyObject *args) {
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

static PyObject * FCEUObject_get_matches(FCEUObject* self, PyObject *args) {
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

static PyMethodDef FCEUObject_methods[] = {
  {"load_game", (PyCFunction)FCEUObject_load_game, METH_VARARGS, "Load NES game."},
  { "load_state", (PyCFunction)FCEUObject_load_state, METH_VARARGS, "Load NES savestate." },
  { "close_game", (PyCFunction)FCEUObject_close_game, METH_VARARGS, "Close loaded NES game." },
  { "reset", (PyCFunction)FCEUObject_reset, METH_VARARGS, "Resets the NES." },
  { "save_state", (PyCFunction)FCEUObject_save_state, METH_VARARGS, "Save NES savestate." },
  { "emulate_frame", (PyCFunction)FCEUObject_emulate_frame, METH_VARARGS, "Emulate a single frame." },
  { "read_memory", (PyCFunction)FCEUObject_read_memory, METH_VARARGS, "Read the value at the given memory address." },
  { "init_memory_search", (PyCFunction)FCEUObject_init_memory_search, METH_VARARGS, "Initialize a search of memory values." },
  { "match_unchanged", (PyCFunction)FCEUObject_match_unchanged, METH_VARARGS, "Matches memory values unchanged since the last search operation." },
  { "match_changed", (PyCFunction)FCEUObject_match_changed, METH_VARARGS, "Matches memory values changed since the last search operation." },
  { "match_equals", (PyCFunction)FCEUObject_match_equals, METH_VARARGS, "Matches memory values equal to a target value." },
  { "get_matches", (PyCFunction)FCEUObject_get_matches, METH_VARARGS, "Get the results of the memory search." },
  {NULL}  /* Sentinel */
};

static PyTypeObject FCEUType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "fceu.FCEU",               /* tp_name */
  sizeof(FCEUObject),        /* tp_basicsize */
  0,                         /* tp_itemsize */
  (destructor)FCEUObject_dealloc, /* tp_dealloc */
  0,                         /* tp_print */
  0,                         /* tp_getattr */
  0,                         /* tp_setattr */
  0,                         /* tp_compare */
  0,                         /* tp_repr */
  0,                         /* tp_as_number */
  0,                         /* tp_as_sequence */
  0,                         /* tp_as_mapping */
  0,                         /* tp_hash */
  0,                         /* tp_call */
  0,                         /* tp_str */
  0,                         /* tp_getattro */
  0,                         /* tp_setattro */
  0,                         /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,        /* tp_flags */
  "FCEU object",             /* tp_doc */
  0,                         /* tp_traverse */
  0,                         /* tp_clear */
  0,                         /* tp_richcompare */
  0,                         /* tp_weaklistoffset */
  0,                         /* tp_iter */
  0,                         /* tp_iternext */
  FCEUObject_methods,        /* tp_methods */
  0,                         /* tp_members */
  0,                         /* tp_getset */
  0,                         /* tp_base */
  0,                         /* tp_dict */
  0,                         /* tp_descr_get */
  0,                         /* tp_descr_set */
  0,                         /* tp_dictoffset */
  (initproc)FCEUObject_init, /* tp_init */
  0,                         /* tp_alloc */
  0,                         /* tp_new */
};
