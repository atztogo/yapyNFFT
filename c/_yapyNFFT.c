#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <numpy/arrayobject.h>
#include "nfft3.h"

#if (PY_MAJOR_VERSION < 3) && (PY_MINOR_VERSION < 6)
#define PYUNICODE_FROMSTRING PyString_FromString
#else
#define PYUNICODE_FROMSTRING PyUnicode_FromString
#endif

static PyObject * py_nfft_init_guru(PyObject *self, PyObject *args);
static PyObject * py_nfft_finalize(PyObject *self, PyObject *args);
static PyObject * py_nfft_precompute_one_psi(PyObject *self, PyObject *args);
static PyObject * py_nfft_trafo(PyObject *self, PyObject *args);
static PyObject * py_nfft_set(PyObject *self, PyObject *args);

struct module_state {
  PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyObject *
error_out(PyObject *m) {
  struct module_state *st = GETSTATE(m);
  PyErr_SetString(st->error, "something bad happened");
  return NULL;
}

static PyMethodDef _yapyNFFT_methods[] = {
  {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},
  {"nfft_init_guru", py_nfft_init_guru, METH_VARARGS,
   "nfft_init_guru"},
  {"nfft_finalize", py_nfft_finalize, METH_VARARGS,
   "nfft_finalize"},
  {"nfft_precompute_one_psi", py_nfft_precompute_one_psi, METH_VARARGS,
   "nfft_precompute_one_psi"},
  {"nfft_trafo", py_nfft_trafo, METH_VARARGS,
   "nfft_trafo"},
  {"nfft_set", py_nfft_set, METH_VARARGS,
   "nfft_set"},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int _yapyNFFT_traverse(PyObject *m, visitproc visit, void *arg) {
  Py_VISIT(GETSTATE(m)->error);
  return 0;
}

static int _yapyNFFT_clear(PyObject *m) {
  Py_CLEAR(GETSTATE(m)->error);
  return 0;
}

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "_yapyNFFT",
  NULL,
  sizeof(struct module_state),
  _yapyNFFT_methods,
  NULL,
  _yapyNFFT_traverse,
  _yapyNFFT_clear,
  NULL
};

#define INITERROR return NULL

PyObject *
PyInit__yapyNFFT(void)

#else
#define INITERROR return

  void
  init_yapyNFFT(void)
#endif
{
  struct module_state *st;
#if PY_MAJOR_VERSION >= 3
  PyObject *module = PyModule_Create(&moduledef);
#else
  PyObject *module = Py_InitModule("_yapyNFFT", _yapyNFFT_methods);
#endif

  if (module == NULL)
    INITERROR;

  st = GETSTATE(module);

  st->error = PyErr_NewException("_yapyNFFT.Error", NULL, NULL);
  if (st->error == NULL) {
    Py_DECREF(module);
    INITERROR;
  }

#if PY_MAJOR_VERSION >= 3
  return module;
#endif
}

static nfft_plan p;

static PyObject * py_nfft_init_guru(PyObject *self, PyObject *args)
{
  int ndim;
  double cutoff;
  PyArrayObject* py_dims_x;
  PyArrayObject* py_dims_f_hat;

  int i;
  int *dims_x, *dims_f_hat;
  int nnode_x;

  if (!PyArg_ParseTuple(args, "iOOd",
			&ndim,
			&py_dims_x,
			&py_dims_f_hat,
                        &cutoff)) {
    return NULL;
  }

  x = (double*)PyArray_DATA(py_x);
  f_hat = (double*)PyArray_DATA(py_f_hat);
  dims_x = (int*)PyArray_DATA(py_dims_x);
  dims_f_hat = (int*)PyArray_DATA(py_dims_f_hat);

  nnode_x = 1;
  for (i = 0; i < ndim; i++) {
    nnode_x *= dims_x[i];
  }

  nfft_init_guru(&p, ndim, dims_x, nnode_x, dims_f_hat, cutoff,
		 PRE_PHI_HUT| PRE_PSI| MALLOC_F_HAT| MALLOC_X| MALLOC_F |
		 FFTW_INIT| FFT_OUT_OF_PLACE,
		 FFTW_ESTIMATE| FFTW_DESTROY_INPUT);

  return Py_RETURN_NONE;
}

static PyObject * py_nfft_finalize(PyObject *self, PyObject *args)
{
  int plan_id;

  if (!PyArg_ParseTuple(args, "i", plan_id)) {
    return NULL;
  }

  nfft_finalize(&p);

  return Py_RETURN_NONE;
}

static PyObject * py_nfft_precompute_one_psi(PyObject *self, PyObject *args)
{
  nfft_precompute_one_psi(&p);

  return Py_RETURN_NONE;
}

static PyObject * py_nfft_trafo(PyObject *self, PyObject *args)
{
  nfft_trafo(&p);

  return Py_RETURN_NONE;
}

static PyObject * py_nfft_set(PyObject *self, PyObject *args)
{
  int ndim;
  PyArrayObject* py_dims_x;
  PyArrayObject* py_dims_f_hat;

  int i, j;
  int *dims_x, *dims_f_hat;
  int nnode_x, nnode_f_hat;
  double *x, *f_hat;

  if (!PyArg_ParseTuple(args, "OOiOOd",
                        &py_x,
                        &py_f_hat,
			&ndim,
			&py_dims_x,
			&py_dims_f_hat)) {
    return NULL;
  }

  x = (double*)PyArray_DATA(py_x);
  f_hat = (double*)PyArray_DATA(py_f_hat);
  dims_x = (int*)PyArray_DATA(py_dims_x);
  dims_f_hat = (int*)PyArray_DATA(py_dims_f_hat);

  nnode_x = 1;
  nnode_f_hat = 1;
  for (i = 0; i < ndim; i++) {
    nnode_x *= dims_x[i];
    nnode_f_hat *= dims_f_hat[i];
  }

  for (i = 0; i < nnode_x; i++) {
    for (j = 0; j < ndim; j++) {
      p.x[i * ndim + j] = x[i * ndim + j];
    }
  }

  for (i = 0; i < nnode_f_hat; i++) {
    p.f_hat[i][0] = f_hat[i * 2];
    p.f_hat[i][1] = f_hat[i * 2 + 1];
  }

  return Py_RETURN_NONE;
}
