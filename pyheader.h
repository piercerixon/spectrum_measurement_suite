#ifndef PYHEADER_H
#define PYHEADER_H

	#ifndef _DEBUG

	#include <Python.h>

	//Python functions
	int init_py_plot();
	PyObject* init_py_function(char*, char*);

	#endif


#endif