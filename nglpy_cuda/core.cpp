#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "ngl_cuda.h"

static PyObject* nglpy_cuda_core_get_edge_list(PyObject *self, PyObject *args) {
    int N;
    int K;

    PyArrayObject *edges_arr;
    PyArrayObject *distances_arr;
    PyArrayObject *indices_arr = NULL;
    if (!PyArg_ParseTuple(args, "O&O&|O&", PyArray_Converter, &edges_arr, PyArray_Converter, &distances_arr, PyArray_Converter, &indices_arr))
        return NULL;

    npy_intp idx[2];
    idx[0] = idx[1] = 0;
    int *edges = (int *)PyArray_GetPtr(edges_arr, idx);
    float *distances = (float *)PyArray_GetPtr(distances_arr, idx);
    int *indices = NULL;

    if (indices_arr != NULL ) {
        indices = (int *)PyArray_GetPtr(indices_arr, idx);
    }

    N = PyArray_DIM(edges_arr, 0);
    K = PyArray_DIM(edges_arr, 1);

    //Do this in two cycles
    //First cycle: grab the total number of edges needed
    int i, k;
    int edge_count = 0;
    for(i = 0; i < N; i++) {
        for(k = 0; k < K; k++) {
            if (edges[i*K+k] != -1) {
	        edge_count++;
            }
        }
    }
    //Second cycle: fill in the array
    PyObject* edge_list = PyList_New(2*edge_count);
    edge_count = 0;
    for(i = 0; i < N; i++) {
        int pi = indices != NULL ? indices[i] : i;
        for(k = 0; k < K; k++) {
            if (edges[i*K+k] != -1) {
	            PyObject* item = Py_BuildValue("(iif)", pi, edges[i*K+k], distances[i*K+k]);
                PyList_SetItem(edge_list, edge_count, item);
	            edge_count++;
	            PyObject* item2 = Py_BuildValue("(iif)", edges[i*K+k], pi, distances[i*K+k]);
                PyList_SetItem(edge_list, edge_count, item2);
	            edge_count++;
            }
        }
    }
    Py_DECREF(edges_arr);
    Py_DECREF(distances_arr);
    return edge_list;
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

static PyObject* nglpy_cuda_core_prune(PyObject *self, PyObject *args, PyObject* kwargs) {
    //import_array();

    int N;
    int D;
    int M;
    int K;
    float lp = 2.0;
    float beta = 1.0;
    bool relaxed = false;
    int count = -1;
    PyArrayObject *X_arr;
    PyArrayObject *edges_arr;
    PyArrayObject *indices_arr = NULL;
    PyArrayObject *template_arr = NULL;
    int steps = -1;

    static char* argnames[] = {"X", "edges", "indices", "template", "steps", "count", "relaxed", "beta", "lp", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&|O&O&iiiff", argnames,
                                     PyArray_Converter, &X_arr,
                                     PyArray_Converter, &edges_arr,
                                     PyArray_Converter, &indices_arr,
                                     PyArray_Converter, &template_arr,
                                     &steps,
                                     &count,
                                     &relaxed,
                                     &beta,
                                     &lp))
        return NULL;

    npy_intp idx[2];
    idx[0] = idx[1] = 0;
    float *X = (float *)PyArray_GetPtr(X_arr, idx);
    int *edges = (int *)PyArray_GetPtr(edges_arr, idx);
    int *indices = NULL;
    if (indices_arr != NULL) {
        indices = (int *)PyArray_GetPtr(indices_arr, idx);
    }

    N = PyArray_DIM(X_arr, 0);
    if (count < 0) {
        count = N;
    }
    D = PyArray_DIM(X_arr, 1);
    M = PyArray_DIM(edges_arr, 0);
    K = PyArray_DIM(edges_arr, 1);

    if (template_arr != NULL) {
        float *erTemplate = (float *)PyArray_GetPtr(template_arr, idx);
        steps = PyArray_DIM(template_arr, 0);
        nglcu::prune_discrete(X, edges, indices, N, D, M, K, erTemplate, steps, relaxed, beta, lp, count);
        Py_DECREF(template_arr);
    }
    else if (steps > 0) {
        nglcu::prune_discrete(X, edges, indices, N, D, M, K, NULL, steps, relaxed, beta, lp, count);
    }
    else {
        nglcu::prune(X, edges, indices, N, D, M, K, relaxed, beta, lp, count);
    }

    Py_DECREF(X_arr);
    //Py_DECREF(edges_arr);

    return PyArray_Return(edges_arr);
}

static PyObject* nglpy_cuda_core_probability(PyObject *self, PyObject *args, PyObject* kwargs) {
    //import_array();
    int N;
    int D;
    int M;
    int K;
    float lp = 2.0;
    float beta = 1.0;
    bool relaxed = false;
    int count = -1;
    PyArrayObject *X_arr;
    PyArrayObject *edges_arr;
    PyArrayObject *probability_arr;
    PyArrayObject *indices_arr = NULL;
    PyArrayObject *template_arr = NULL;
    int steps = -1;

    float steepness = 3;

    static char* argnames[] = {"X", "edges", "steepness", "indices", "template", "steps", "count", "relaxed", "beta", "lp", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&|fO&O&iipff", argnames,
                                     PyArray_Converter, &X_arr,
                                     PyArray_Converter, &edges_arr,
                                     &steepness,
                                     PyArray_Converter, &indices_arr,
                                     PyArray_Converter, &template_arr,
                                     &steps,
                                     &count,
                                     &relaxed,
                                     &beta,
                                     &lp))
        return NULL;

    npy_intp idx[2];
    idx[0] = idx[1] = 0;
    float *X = (float *)PyArray_GetPtr(X_arr, idx);
    int *edges = (int *)PyArray_GetPtr(edges_arr, idx);
    int *indices = NULL;
    if (indices_arr != NULL) {
        indices = (int *)PyArray_GetPtr(indices_arr, idx);
    }

    N = PyArray_DIM(X_arr, 0);
    if (count < 0) {
        count = N;
    }
    D = PyArray_DIM(X_arr, 1);
    M = PyArray_DIM(edges_arr, 0);
    K = PyArray_DIM(edges_arr, 1);

    //Reuse this array since we already have it allocated
    idx[0] = count;
    idx[1] = K;
    probability_arr = (PyArrayObject *)PyArray_ZEROS(2, idx, NPY_FLOAT32, 0);
    idx[0] = idx[1] = 0;
    float *probabilities = (float *)PyArray_GetPtr(probability_arr, idx);
    nglcu::associate_probability(X, edges, probabilities, indices, N, D, M, K, steepness, relaxed, beta, lp, count);

    Py_DECREF(X_arr);
    //Py_DECREF(edges_arr);

    return PyArray_Return(probability_arr);
}

static PyObject* nglpy_cuda_core_get_available_memory(PyObject *self) {
    return Py_BuildValue("i", (int)nglcu::get_available_device_memory());
}

static PyMethodDef nglpy_cuda_core_methods[] = {
    {"get_edge_list", (PyCFunction)nglpy_cuda_core_get_edge_list, METH_VARARGS, ""},
    {"create_template",(PyCFunction)nglpy_cuda_core_create_template, METH_VARARGS, ""},
    {"min_distance_from_edge",(PyCFunction)nglpy_cuda_core_min_distance_from_edge, METH_VARARGS, ""},
    {"prune",(PyCFunction)nglpy_cuda_core_prune, METH_VARARGS|METH_KEYWORDS, ""},
    {"associate_probability",(PyCFunction)nglpy_cuda_core_probability, METH_VARARGS|METH_KEYWORDS, ""},
    {"get_available_device_memory",(PyCFunction)nglpy_cuda_core_get_available_memory, METH_NOARGS, ""},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
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
#else
PyMODINIT_FUNC initcore(){
    import_array();
    Py_InitModule("core", nglpy_cuda_core_methods);
}
#endif
