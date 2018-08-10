#include "Python.h"
#include "ngl_cuda.h"

static PyObject* nglcu_core_get_edge_list(PyObject *self, PyObject *args){
    int *edges;
    int N;
    int K;
		if (!PyArg_ParseTuple(args, "ii", &edges, &N, &K))return NULL;
		return Py_BuildValue("i", nglcu::add(edges, N, K));
	}

static PyObject* pipcudemo_core_subtract(PyObject *self, PyObject *args){
	int a,b;
		if (!PyArg_ParseTuple(args, "ii", &a, &b))return NULL;
		return Py_BuildValue("i", pipcudemo::subtract(a, b));
	}

static PyObject* pipcudemo_core_multiply(PyObject *self, PyObject *args){
	int a,b;
		if (!PyArg_ParseTuple(args, "ii", &a, &b))return NULL;
		return Py_BuildValue("i", pipcudemo::multiply(a, b));
	}

static PyObject* pipcudemo_core_divide(PyObject *self, PyObject *args){
	int a,b;
		if (!PyArg_ParseTuple(args, "ii", &a, &b))return NULL;
		return Py_BuildValue("i", pipcudemo::divide(a, b));
	}

static PyMethodDef pipcudemo_core_methods[] = {
	{"add",(PyCFunction)pipcudemo_core_add,	METH_VARARGS, "Add two integers"},
	{"subtract",(PyCFunction)pipcudemo_core_subtract, METH_VARARGS, "Subtract two integers"},
	{"multiply",(PyCFunction)pipcudemo_core_multiply, METH_VARARGS, "Multiply two integers"},
	{"divide",(PyCFunction)pipcudemo_core_divide,     METH_VARARGS, "Divide two integers"},
   	{NULL, NULL, 0, NULL}
};


static struct PyModuleDef module_def = {
	PyModuleDef_HEAD_INIT,
	"pipcudemo.core",
	"An example project showing how to build a pip-installable Python package that invokes custom CUDA/C++ code.",
	-1,
	pipcudemo_core_methods
};

PyMODINIT_FUNC PyInit_core(){
	return PyModule_Create(&module_def);
}