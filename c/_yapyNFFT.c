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

static PyObject * py_nfft_init(PyObject *self, PyObject *args);
static PyObject * py_nfft_init_guru(PyObject *self, PyObject *args);
static PyObject * py_nfft_init_nd(PyObject *self, PyObject *args);
static PyObject * py_nfft_finalize(PyObject *self, PyObject *args);
static PyObject * py_nfft_precompute_one_psi(PyObject *self, PyObject *args);
static PyObject * py_nfft_trafo(PyObject *self, PyObject *args);
static PyObject * py_nfft_trafo_direct(PyObject *self, PyObject *args);
static PyObject * py_nfft_adjoint(PyObject *self, PyObject *args);
static PyObject * py_nfft_adjoint_direct(PyObject *self, PyObject *args);
static PyObject * py_nfft_set(PyObject *self, PyObject *args);
static PyObject * py_nfft_get_f(PyObject *self, PyObject *args);
static PyObject * py_nfft_get_f_hat(PyObject *self, PyObject *args);
static PyObject * py_nfft_get_N(PyObject *self, PyObject *args);
static PyObject * py_nfft_get_M(PyObject *self, PyObject *args);

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
  {"nfft_init", py_nfft_init, METH_VARARGS,
   "nfft_init"},
  {"nfft_init_guru", py_nfft_init_guru, METH_VARARGS,
   "nfft_init_guru"},
  {"nfft_init_nd", py_nfft_init_nd, METH_VARARGS,
   "nfft_init_nd"},
  {"nfft_finalize", py_nfft_finalize, METH_VARARGS,
   "nfft_finalize"},
  {"nfft_precompute_one_psi", py_nfft_precompute_one_psi, METH_VARARGS,
   "nfft_precompute_one_psi"},
  {"nfft_trafo", py_nfft_trafo, METH_VARARGS,
   "nfft_trafo"},
  {"nfft_trafo_direct", py_nfft_trafo_direct, METH_VARARGS,
   "nfft_trafo_direct"},
  {"nfft_adjoint", py_nfft_adjoint, METH_VARARGS,
   "nfft_adjoint"},
  {"nfft_adjoint_direct", py_nfft_adjoint_direct, METH_VARARGS,
   "nfft_adjoint_direct"},
  {"nfft_set", py_nfft_set, METH_VARARGS,
   "nfft_set"},
  {"nfft_get_f", py_nfft_get_f, METH_VARARGS,
   "nfft_get_f"},
  {"nfft_get_f_hat", py_nfft_get_f_hat, METH_VARARGS,
   "nfft_get_f_hat"},
  {"nfft_get_N", py_nfft_get_N, METH_VARARGS,
   "nfft_get_N"},
  {"nfft_get_M", py_nfft_get_M, METH_VARARGS,
   "nfft_get_M"},
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

static PyObject * py_nfft_init(PyObject *self, PyObject *args)
{
  int nnode;
  PyArrayObject* py_dims_N;

  int ndim;
  int *dims_N;

  if (!PyArg_ParseTuple(args, "Oi",
                        &py_dims_N, /* coefficient grid */
                        &nnode)) { /* sampling points */
    return NULL;
  }

  ndim = PyArray_DIM(py_dims_N, 0);
  dims_N = (int*)PyArray_DATA(py_dims_N);

  nfft_init(&p, ndim, dims_N, nnode);

  Py_RETURN_NONE;
}

static PyObject * py_nfft_init_guru(PyObject *self, PyObject *args)
{
  int nnode, cutoff;
  PyArrayObject* py_dims_N;
  PyArrayObject* py_dims_n;

  int ndim;
  int *dims_N, *dims_n;

  if (!PyArg_ParseTuple(args, "OOii",
                        &py_dims_N, /* coefficient grid */
                        &py_dims_n, /* over sampling grid */
                        &nnode, /* sampling points */
                        &cutoff)) {
    return NULL;
  }

  ndim = PyArray_DIM(py_dims_N, 0);
  dims_N = (int*)PyArray_DATA(py_dims_N);
  dims_n = (int*)PyArray_DATA(py_dims_n);

  nfft_init_guru(&p, ndim, dims_N, nnode, dims_n, cutoff,
                 PRE_PHI_HUT| PRE_PSI| MALLOC_F_HAT| MALLOC_X| MALLOC_F |
                 FFTW_INIT| FFT_OUT_OF_PLACE,
                 FFTW_ESTIMATE| FFTW_DESTROY_INPUT);

  Py_RETURN_NONE;
}

static PyObject * py_nfft_init_nd(PyObject *self, PyObject *args)
{
  int nnode;
  PyArrayObject* py_dims_N;

  int ndim;
  int *dims_N;

  if (!PyArg_ParseTuple(args, "Oi",
                        &py_dims_N, /* coefficient grid */
                        &nnode)) { /* sampling points */
    return NULL;
  }

  ndim = PyArray_DIM(py_dims_N, 0);
  dims_N = (int*)PyArray_DATA(py_dims_N);

  if (ndim == 1) {
    nfft_init_1d(&p, dims_N[0], nnode);
  }
  if (ndim == 2) {
    nfft_init_2d(&p, dims_N[0], dims_N[1], nnode);
  }
  if (ndim == 3) {
    nfft_init_3d(&p, dims_N[0], dims_N[1], dims_N[2], nnode);
  }

  Py_RETURN_NONE;
}

static PyObject * py_nfft_finalize(PyObject *self, PyObject *args)
{
  int plan_id;

  if (!PyArg_ParseTuple(args, "i", &plan_id)) {
    return NULL;
  }

  nfft_finalize(&p);

  Py_RETURN_NONE;
}

static PyObject * py_nfft_precompute_one_psi(PyObject *self, PyObject *args)
{
  if (p.flags & PRE_ONE_PSI) {
    nfft_precompute_one_psi(&p);
  }

  Py_RETURN_NONE;
}

static PyObject * py_nfft_trafo(PyObject *self, PyObject *args)
{
  nfft_trafo(&p);

  Py_RETURN_NONE;
}

static PyObject * py_nfft_trafo_direct(PyObject *self, PyObject *args)
{
  nfft_trafo_direct(&p);

  Py_RETURN_NONE;
}

static PyObject * py_nfft_adjoint(PyObject *self, PyObject *args)
{
  nfft_adjoint(&p);

  Py_RETURN_NONE;
}

static PyObject * py_nfft_adjoint_direct(PyObject *self, PyObject *args)
{
  nfft_adjoint_direct(&p);

  Py_RETURN_NONE;
}

static PyObject * py_nfft_set(PyObject *self, PyObject *args)
{
  PyArrayObject* py_x;
  PyArrayObject* py_f_hat;
  char *dtype, *order;
  int shift_grid;

  int i, j, k, adrs, adrs_shift;
  double *x, *f_hat;
  npy_intp *dims;

  if (!PyArg_ParseTuple(args, "OOssi",
                        &py_x,
                        &py_f_hat,
                        &dtype,
                        &order,
                        &shift_grid)) {
    return NULL;
  }

  x = (double*)PyArray_DATA(py_x);
  f_hat = (double*)PyArray_DATA(py_f_hat);
  dims = PyArray_DIMS(py_f_hat);

  if (order[0] == 'C') {
    for (i = 0; i < p.M_total * p.d; i++) {
      p.x[i] = x[i];
    }
  } else {
    for (i = 0; i < p.M_total; i++) {
      for (j = 0; j < p.d; j++) {
        p.x[i * p.d + j] = x[i * p.d + p.d - j - 1];
      }
    }
  }

  if (shift_grid) {
    if (dtype[0] == 'c') {
      dims[2] /= 2;
      for (i = 0; i < dims[0]; i++) {
        for (j = 0; j < dims[1]; j++) {
          for (k = 0; k < dims[2]; k++) {
            adrs = i * dims[1] * dims[2] + j * dims[2] + k;
            adrs_shift = (((i + dims[0] / 2) % dims[0]) * dims[1] * dims[2] +
                          ((j + dims[1] / 2) % dims[1]) * dims[2] +
                          ((k + dims[2] / 2) % dims[2]));
            p.f_hat[adrs_shift][0] = f_hat[adrs * 2];
            p.f_hat[adrs_shift][1] = f_hat[adrs * 2 + 1];
          }
        }
      }
    } else {
      for (i = 0; i < dims[0]; i++) {
        for (j = 0; j < dims[1]; j++) {
          for (k = 0; k < dims[2]; k++) {
            adrs = i * dims[1] * dims[2] + j * dims[2] + k;
            adrs_shift = (((i + dims[0] / 2) % dims[0]) * dims[1] * dims[2] +
                          ((j + dims[1] / 2) % dims[1]) * dims[2] +
                          ((k + dims[2] / 2) % dims[2]));
            p.f_hat[adrs_shift][0] = f_hat[adrs];
            p.f_hat[adrs_shift][1] = 0;
          }
        }
      }
    }
  } else {
    if (dtype[0] == 'c') {
      for (i = 0; i < p.N_total; i++) {
        p.f_hat[i][0] = f_hat[i * 2];
        p.f_hat[i][1] = f_hat[i * 2 + 1];
      }
    } else {
      for (i = 0; i < p.N_total; i++) {
        p.f_hat[i][0] = f_hat[i];
        p.f_hat[i][1] = 0;
      }
    }
  }

  Py_RETURN_NONE;
}

static PyObject * py_nfft_get_f(PyObject *self, PyObject *args)
{
  PyArrayObject* py_f;

  int i;
  double *f;

  if (!PyArg_ParseTuple(args, "O",
                        &py_f)) {
    return NULL;
  }

  f = (double*)PyArray_DATA(py_f);

  for (i = 0; i < p.M_total; i++) {
    f[i * 2] = p.f[i][0];
    f[i * 2 + 1] = p.f[i][1];
  }

  Py_RETURN_NONE;
}

static PyObject * py_nfft_get_f_hat(PyObject *self, PyObject *args)
{
  PyArrayObject* py_f_hat;

  int i;
  double *f_hat;

  if (!PyArg_ParseTuple(args, "O",
                        &py_f_hat)) {
    return NULL;
  }

  f_hat = (double*)PyArray_DATA(py_f_hat);

  for (i = 0; i < p.N_total; i++) {
    f_hat[i * 2] = p.f_hat[i][0];
    f_hat[i * 2 + 1] = p.f_hat[i][1];
  }

  Py_RETURN_NONE;
}

static PyObject * py_nfft_get_N(PyObject *self, PyObject *args)
{
  int i;
  PyObject *dims_N;

  dims_N = PyList_New(p.d);
  for (i = 0; i < p.d; i++) {
    PyList_SetItem(dims_N, i, PyLong_FromLong((long) p.N[i]));
  }

  return dims_N;
}

static PyObject * py_nfft_get_M(PyObject *self, PyObject *args)
{
  return PyLong_FromLong((long) p.M_total);
}
