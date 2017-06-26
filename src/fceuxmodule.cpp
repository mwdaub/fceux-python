#include <Python.h>
#include <numpy/arrayobject.h>

bool FCEUI_Initialize();

//input device types for the standard joystick port
enum ESI : unsigned int;
void FCEUI_SetInput(int port, ESI type, void *ptr, int attrib);

void FCEU_printf(char *format, ...);

struct FCEUGI;
FCEUGI *FCEUI_LoadGame(const char *name, int OverwriteVidMode, bool silent = false);

typedef struct {
	unsigned char r,g,b;
} pal;
extern pal *palo;
void FCEUI_Emulate(unsigned char **pXBuf, signed int **SoundBuf, signed int *SoundBufSize, int skip);

unsigned int input;

static PyObject * fceux_load_rom(PyObject *self, PyObject *args) {
  const char * input;

  // parse arguments
  if (!PyArg_ParseTuple(args, "s", &input)) {
    return NULL;
  }

  if (!FCEUI_LoadGame(input, 1)) {
    return NULL;
  }
  FCEUI_SetInput(0, 1, &input, 0);

  Py_INCREF(Py_None);
  return Py_None;
}

static long int dims[] = {256, 256, 3};

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
  for (size_t i = 0; i < 256*256; i++) {
    unsigned char idx = gfx[i] - 128;
    data[3*i] = palo[idx].r;
    data[3*i+1] = palo[idx].g;
    data[3*i+2] = palo[idx].b;
    // for (size_t j = 0; j < 3; j++) {
    //   data[i*3 + j] = gfx[j];
    // }
    // gfx++;
  }
  return array;
}

static PyMethodDef FceuxMethods[] = {
  { "load_rom", fceux_load_rom, METH_VARARGS, "Load NES rom" },
  { "emulate_frame", fceux_emulate_frame, METH_VARARGS, "Emulate a single frame" },
  { NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC initfceux(void)
{
  (void) Py_InitModule("fceux", FceuxMethods);
  import_array();
  FCEUI_Initialize();
}
