#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "ngl_cuda.h"
#include <iostream>

static PyObject* nglpy_cuda_core_get_edge_list(PyObject *self, PyObject *args) {
    int *edges;
    int N;
    int K;
    PyObject *edges_obj;
    if (!PyArg_ParseTuple(args, "Oii", &edges_obj, &N, &K))
        return NULL;

    edges = new int[N*K];
    PyObject *iter = PyObject_GetIter(edges_obj);
    if (!iter) {
        //error not iterator
    }

    int k=0, n=0;
    while (n*K+k < N*K) {
        PyObject *next = PyIter_Next(iter);
	if (!next) {
	    break;
	}
	if (!PyFloat_Check(next)) {
	}

	edges[n*K+k++] = PyFloat_AsDouble(next);
	if(k == K) {
	    k=0;
	    n++;
	}
    }
    std::vector< std::pair<int, int> > edge_list = nglcu::get_edge_list(edges, N, K);
    delete [] edges;
    return Py_BuildValue("O", edge_list);
}

static PyObject* nglpy_cuda_core_create_template(PyObject *self, PyObject *args) {
    float beta;
    int p;
    int steps;
    if (!PyArg_ParseTuple(args, "fii", &beta, &p, &steps))
        return NULL;
    float data[steps];
    nglcu::create_template(data, beta, p, steps);

    PyObject* template_value_list = PyList_New(steps);
    for (int i = 0; i < steps; i++) {
	PyObject* item = Py_BuildValue("f", data[i]);
        PyList_SetItem(template_value_list, i, item);
    }
    return template_value_list;
}

static PyObject* nglpy_cuda_core_min_distance_from_edge(PyObject *self, PyObject *args) {
    float t;
    float beta;
    float p;
    if (!PyArg_ParseTuple(args, "fff", &t, &beta, &p))
        return NULL;
    return Py_BuildValue("f", nglcu::min_distance_from_edge(t, beta, p));
}

static PyObject* nglpy_cuda_core_prune_discrete(PyObject *self, PyObject *args) {
    int N;
    int D;
    int K;
    int steps;

    // Two signatures for this method where either collection below is specified:
    float beta;
    float p;
    // OR
    float *erTemplate;
    
    PyArrayObject *X_arr;
    PyArrayObject *edges_arr;
    PyArrayObject *template_arr;

    npy_intp idx[2];
    idx[0] = idx[1] = 0;

    if(PyArg_ParseTuple(args, "iiiiffO&O&", &N, &D, &K, &steps, &beta, &p, PyArray_Converter, &X_arr, PyArray_Converter, &edges_arr)) {
        float *X = (float *)PyArray_GetPtr(X_arr, idx);
        int *edges = (int *)PyArray_GetPtr(edges_arr, idx);
        nglcu::prune_discrete(N, D, K, steps, beta, p, X, edges);
        Py_DECREF(X_arr);
        return PyArray_Return(edges_arr);
    }
    else if (PyArg_ParseTuple(args, "iiiiO&O&O&", &N, &D, &K, &steps, PyArray_Converter, &template_arr, PyArray_Converter, &X_arr, PyArray_Converter, &edges_arr)) {
        // The fact that we passed through the first if clause means the error 
	// indicator will be set with the following error message:
	// TypeError: function takes exactly 8 arguments (7 given)
	// By clearing this indicator, we are allowing the function to act in
	// an overloaded fashion. TODO: I should probably verify that the
	// error indicator is a PyExc_TypeErro.r
	PyErr_Clear();
       	float *X = (float *)PyArray_GetPtr(X_arr, idx);
        int *edges = (int *)PyArray_GetPtr(edges_arr, idx);
        float *erTemplate = (float *)PyArray_GetPtr(template_arr, idx);
        nglcu::prune_discrete(N, D, K, steps, erTemplate, X, edges);
        Py_DECREF(X_arr);
        Py_DECREF(template_arr);
        return PyArray_Return(edges_arr);
    }
    else {
        return NULL;
    }
}

static PyObject* nglpy_cuda_core_prune(PyObject *self, PyObject *args) {
    //import_array();
    int N;
    int D;
    int K;
    float lp;
    float beta;
    PyArrayObject *X_arr;
    PyArrayObject *edges_arr;
    if (!PyArg_ParseTuple(args, "iiiffO&O&", &N, &D, &K, &lp, &beta, PyArray_Converter, &X_arr, PyArray_Converter, &edges_arr))
        return NULL;

    npy_intp idx[2];
    idx[0] = idx[1] = 0;
    float *X = (float *)PyArray_GetPtr(X_arr, idx);
    int *edges = (int *)PyArray_GetPtr(edges_arr, idx);

    nglcu::prune(N, D, K, lp, beta, X, edges);
    Py_DECREF(X_arr);
    //Py_DECREF(edges_arr);

    return PyArray_Return(edges_arr);
}

static PyMethodDef nglpy_cuda_core_methods[] = {
    {"get_edge_list", (PyCFunction)nglpy_cuda_core_get_edge_list, METH_VARARGS, ""},
    {"create_template",(PyCFunction)nglpy_cuda_core_create_template, METH_VARARGS, ""},
    {"min_distance_from_edge",(PyCFunction)nglpy_cuda_core_min_distance_from_edge, METH_VARARGS, ""},
    {"prune_discrete",(PyCFunction)nglpy_cuda_core_prune_discrete, METH_VARARGS, ""},
    {"prune",(PyCFunction)nglpy_cuda_core_prune, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "nglpy_cuda.core",
    "A Python wrapper to a CUDA-based implementation of the Neighborhood Graph Library (NGL).",
    -1,
    nglpy_cuda_core_methods
};

PyMODINIT_FUNC PyInit_core(){
    import_array();
    return PyModule_Create(&module_def);
}


