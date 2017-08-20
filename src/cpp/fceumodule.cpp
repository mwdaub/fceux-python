#include <Python.h>
#include <numpy/arrayobject.h>

#include "fceuobject.h"

// List of fceu module methods.
static PyMethodDef fceu_methods[] = {
  { NULL, NULL, 0, NULL }
};

// Initialize the fceu module.
PyMODINIT_FUNC initfceu(void) {
  PyObject *m;

  FCEUType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&FCEUType) < 0)
    return;

  m = Py_InitModule("fceu", fceu_methods);
  if (m == NULL) {
    return;
  }

  Py_INCREF(&FCEUType);
  PyModule_AddObject(m, "FCEU", (PyObject *)&FCEUType);

  InvalidRomError = PyErr_NewException("fceu.invalid_rom", NULL, NULL);
  Py_INCREF(InvalidRomError);
  PyModule_AddObject(m, "error", InvalidRomError);

  IllegalStateError = PyErr_NewException("fceu.illegal_state", NULL, NULL);
  Py_INCREF(IllegalStateError);
  PyModule_AddObject(m, "error", IllegalStateError);

  import_array();
}
