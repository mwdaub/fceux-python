#include <stdio.h>

#include "fceuxmodule.h"

// Python error for when a file is not a valid ROM format.
static PyObject *InvalidRomError;

// Game input.
// "A" = 1, "B" = 2, "SELECT" = 4, "START" = 8, "UP" = 16, "DOWN" = 32, "LEFT" = 64, "RIGHT" = 128
// Combinations of buttons are indicated by taking their sums.
// To specify input for player 2, take the identical player 1 input and bit shift left by 8.
unsigned int input;

static PyObject * fceux_load_rom(PyObject *self, PyObject *args) {
  const char * arg;

  // parse arguments
  if (!PyArg_ParseTuple(args, "s", &arg)) {
    return NULL;
  }

  if (!FCEUI_LoadGame(arg, 1)) {
    char *msg;
    asprintf(&msg,"File %s is not a valid ROM or does not exist.", arg);
    PyErr_SetString(InvalidRomError, msg);
    return NULL;
  }
  FCEUI_SetInput(0, SI_GAMEPAD, &input, 0);

  Py_INCREF(Py_None);
  return Py_None;
}

// Dimensions of the game image.
static long int dims[] = {240, 256, 3};

static PyObject * fceux_emulate_frame(PyObject *self, PyObject *args) {
  // parse arguments
  if (!PyArg_ParseTuple(args, "I", &input)) {
    return NULL;
  }

  unsigned char *gfx;
  signed int *sound;
  signed int ssize;
  FCEUI_Emulate(&gfx, &sound, &ssize, 0);

  PyObject *array = PyArray_SimpleNew(3, dims, NPY_UINT8);
  unsigned char *data = (unsigned char *) PyArray_DATA(array);
  for (size_t i = 0; i < 240*256; i++) {
    unsigned char idx = gfx[i] - 128;
    data[3*i] = palo[idx].r;
    data[3*i+1] = palo[idx].g;
    data[3*i+2] = palo[idx].b;
  }
  return array;
}

static PyObject * fceux_read_memory(PyObject *self, PyObject *args) {
  unsigned short arg;

  // parse arguments
  if (!PyArg_ParseTuple(args, "H", &arg)) {
    return NULL;
  }

  // read memory
  unsigned char val = GetMem(arg);
  return Py_BuildValue("I", val);
}

static PyMethodDef FceuxMethods[] = {
  { "load_rom", fceux_load_rom, METH_VARARGS, "Load NES rom" },
  { "emulate_frame", fceux_emulate_frame, METH_VARARGS, "Emulate a single frame" },
  { "read_memory", fceux_read_memory, METH_VARARGS, "Read the value at the given memory address" },
  { NULL, NULL, 0, NULL }
};

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
