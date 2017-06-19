#include <Python.h>

static PyObject * fceux_hello(PyObject *self, PyObject *args)
{
  const char * input;
  PyObject * ret;

  // parse arguments
  if (!PyArg_ParseTuple(args, "s", &input)) {
    return NULL;
  }

  // build the resulting string into a Python object.
  ret = PyString_FromString(input);
                
  return ret;
}

static PyMethodDef FceuxMethods[] = {
  { "hello", fceux_hello, METH_VARARGS, "Say hello" },
  { NULL, NULL, 0, NULL }
};

// DL_EXPORT(void)
PyMODINIT_FUNC initfceux(void)
{
  (void) Py_InitModule("fceux", FceuxMethods);
}
