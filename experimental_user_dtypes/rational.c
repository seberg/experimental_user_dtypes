/* Fixed size rational numbers exposed to Python */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <numpy/npy_3kcompat.h>
#include <math.h>

/* For testing experimental dtype API */
#include "numpy/experimental_dtype_api.h"

/* copied over from private common.h */
#define error_converting(x)  (((x) == -1) && PyErr_Occurred())

/* Relevant arithmetic exceptions */

/* Uncomment the following line to work around a bug in numpy */
/* #define ACQUIRE_GIL */

static void
set_overflow(void) {
#ifdef ACQUIRE_GIL
    /* Need to grab the GIL to dodge a bug in numpy */
    PyGILState_STATE state = PyGILState_Ensure();
#endif
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_OverflowError,
                "overflow in rational arithmetic");
    }
#ifdef ACQUIRE_GIL
    PyGILState_Release(state);
#endif
}

static void
set_zero_divide(void) {
#ifdef ACQUIRE_GIL
    /* Need to grab the GIL to dodge a bug in numpy */
    PyGILState_STATE state = PyGILState_Ensure();
#endif
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_ZeroDivisionError,
                "zero divide in rational arithmetic");
    }
#ifdef ACQUIRE_GIL
    PyGILState_Release(state);
#endif
}

/* Integer arithmetic utilities */

static NPY_INLINE npy_int32
safe_neg(npy_int32 x) {
    if (x==(npy_int32)1<<31) {
        set_overflow();
    }
    return -x;
}

static NPY_INLINE npy_int32
safe_abs32(npy_int32 x) {
    npy_int32 nx;
    if (x>=0) {
        return x;
    }
    nx = -x;
    if (nx<0) {
        set_overflow();
    }
    return nx;
}

static NPY_INLINE npy_int64
safe_abs64(npy_int64 x) {
    npy_int64 nx;
    if (x>=0) {
        return x;
    }
    nx = -x;
    if (nx<0) {
        set_overflow();
    }
    return nx;
}

static NPY_INLINE npy_int64
gcd(npy_int64 x, npy_int64 y) {
    x = safe_abs64(x);
    y = safe_abs64(y);
    if (x < y) {
        npy_int64 t = x;
        x = y;
        y = t;
    }
    while (y) {
        npy_int64 t;
        x = x%y;
        t = x;
        x = y;
        y = t;
    }
    return x;
}

static NPY_INLINE npy_int64
lcm(npy_int64 x, npy_int64 y) {
    npy_int64 lcm;
    if (!x || !y) {
        return 0;
    }
    x /= gcd(x,y);
    lcm = x*y;
    if (lcm/y!=x) {
        set_overflow();
    }
    return safe_abs64(lcm);
}

/* Fixed precision rational numbers */

typedef struct {
    /* numerator */
    npy_int32 n;
    /*
     * denominator minus one: numpy.zeros() uses memset(0) for non-object
     * types, so need to ensure that rational(0) has all zero bytes
     */
    npy_int32 dmm;
} rational;

static NPY_INLINE rational
make_rational_int(npy_int64 n) {
    rational r = {(npy_int32)n,0};
    if (r.n != n) {
        set_overflow();
    }
    return r;
}

static rational
make_rational_slow(npy_int64 n_, npy_int64 d_) {
    rational r = {0};
    if (!d_) {
        set_zero_divide();
    }
    else {
        npy_int64 g = gcd(n_,d_);
        npy_int32 d;
        n_ /= g;
        d_ /= g;
        r.n = (npy_int32)n_;
        d = (npy_int32)d_;
        if (r.n!=n_ || d!=d_) {
            set_overflow();
        }
        else {
            if (d <= 0) {
                d = -d;
                r.n = safe_neg(r.n);
            }
            r.dmm = d-1;
        }
    }
    return r;
}

static NPY_INLINE npy_int32
d(rational r) {
    return r.dmm+1;
}

/* Assumes d_ > 0 */
static rational
make_rational_fast(npy_int64 n_, npy_int64 d_) {
    npy_int64 g = gcd(n_,d_);
    rational r;
    n_ /= g;
    d_ /= g;
    r.n = (npy_int32)n_;
    r.dmm = (npy_int32)(d_-1);
    if (r.n!=n_ || r.dmm+1!=d_) {
        set_overflow();
    }
    return r;
}

static NPY_INLINE rational
rational_negative(rational r) {
    rational x;
    x.n = safe_neg(r.n);
    x.dmm = r.dmm;
    return x;
}

static NPY_INLINE rational
rational_add(rational x, rational y) {
    /*
     * Note that the numerator computation can never overflow int128_t,
     * since each term is strictly under 2**128/4 (since d > 0).
     */
    return make_rational_fast((npy_int64)x.n*d(y)+(npy_int64)d(x)*y.n,
        (npy_int64)d(x)*d(y));
}

static NPY_INLINE rational
rational_subtract(rational x, rational y) {
    /* We're safe from overflow as with + */
    return make_rational_fast((npy_int64)x.n*d(y)-(npy_int64)d(x)*y.n,
        (npy_int64)d(x)*d(y));
}

static NPY_INLINE rational
rational_multiply(rational x, rational y) {
    /* We're safe from overflow as with + */
    return make_rational_fast((npy_int64)x.n*y.n,(npy_int64)d(x)*d(y));
}

static NPY_INLINE rational
rational_divide(rational x, rational y) {
    return make_rational_slow((npy_int64)x.n*d(y),(npy_int64)d(x)*y.n);
}

static NPY_INLINE npy_int64
rational_floor(rational x) {
    /* Always round down */
    if (x.n>=0) {
        return x.n/d(x);
    }
    /*
     * This can be done without casting up to 64 bits, but it requires
     * working out all the sign cases
     */
    return -((-(npy_int64)x.n+d(x)-1)/d(x));
}

static NPY_INLINE npy_int64
rational_ceil(rational x) {
    return -rational_floor(rational_negative(x));
}

static NPY_INLINE rational
rational_remainder(rational x, rational y) {
    return rational_subtract(x, rational_multiply(y,make_rational_int(
                    rational_floor(rational_divide(x,y)))));
}

static NPY_INLINE rational
rational_abs(rational x) {
    rational y;
    y.n = safe_abs32(x.n);
    y.dmm = x.dmm;
    return y;
}

static NPY_INLINE npy_int64
rational_rint(rational x) {
    /*
     * Round towards nearest integer, moving exact half integers towards
     * zero
     */
    npy_int32 d_ = d(x);
    return (2*(npy_int64)x.n+(x.n<0?-d_:d_))/(2*(npy_int64)d_);
}

static NPY_INLINE int
rational_sign(rational x) {
    return x.n<0?-1:x.n==0?0:1;
}

static NPY_INLINE rational
rational_inverse(rational x) {
    rational y = {0};
    if (!x.n) {
        set_zero_divide();
    }
    else {
        npy_int32 d_;
        y.n = d(x);
        d_ = x.n;
        if (d_ <= 0) {
            d_ = safe_neg(d_);
            y.n = -y.n;
        }
        y.dmm = d_-1;
    }
    return y;
}

static NPY_INLINE int
rational_eq(rational x, rational y) {
    /*
     * Since we enforce d > 0, and store fractions in reduced form,
     * equality is easy.
     */
    return x.n==y.n && x.dmm==y.dmm;
}

static NPY_INLINE int
rational_ne(rational x, rational y) {
    return !rational_eq(x,y);
}

static NPY_INLINE int
rational_lt(rational x, rational y) {
    return (npy_int64)x.n*d(y) < (npy_int64)y.n*d(x);
}

static NPY_INLINE int
rational_gt(rational x, rational y) {
    return rational_lt(y,x);
}

static NPY_INLINE int
rational_le(rational x, rational y) {
    return !rational_lt(y,x);
}

static NPY_INLINE int
rational_ge(rational x, rational y) {
    return !rational_lt(x,y);
}

static NPY_INLINE npy_int32
rational_int(rational x) {
    return x.n/d(x);
}

static NPY_INLINE double
rational_double(rational x) {
    return (double)x.n/d(x);
}

static NPY_INLINE int
rational_nonzero(rational x) {
    return x.n!=0;
}

static int
scan_rational(const char** s, rational* x) {
    long n,d;
    int offset;
    const char* ss;
    if (sscanf(*s,"%ld%n",&n,&offset)<=0) {
        return 0;
    }
    ss = *s+offset;
    if (*ss!='/') {
        *s = ss;
        *x = make_rational_int(n);
        return 1;
    }
    ss++;
    if (sscanf(ss,"%ld%n",&d,&offset)<=0 || d<=0) {
        return 0;
    }
    *s = ss+offset;
    *x = make_rational_slow(n,d);
    return 1;
}

/* Expose rational to Python as a numpy scalar */

typedef struct {
    PyObject_HEAD
    rational r;
} PyRational;

static PyTypeObject PyRational_Type;

static NPY_INLINE int
PyRational_Check(PyObject* object) {
    return PyObject_IsInstance(object,(PyObject*)&PyRational_Type);
}

static PyObject*
PyRational_FromRational(rational x) {
    PyRational* p = (PyRational*)PyRational_Type.tp_alloc(&PyRational_Type,0);
    if (p) {
        p->r = x;
    }
    return (PyObject*)p;
}

static PyObject*
pyrational_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    Py_ssize_t size;
    PyObject* x[2];
    long n[2]={0,1};
    int i;
    rational r;
    if (kwds && PyDict_Size(kwds)) {
        PyErr_SetString(PyExc_TypeError,
                "constructor takes no keyword arguments");
        return 0;
    }
    size = PyTuple_GET_SIZE(args);
    if (size > 2) {
        PyErr_SetString(PyExc_TypeError,
                "expected rational or numerator and optional denominator");
        return 0;
    }

    if (size == 1) {
        x[0] = PyTuple_GET_ITEM(args, 0);
        if (PyRational_Check(x[0])) {
            Py_INCREF(x[0]);
            return x[0];
        }
        // TODO: allow construction from unicode strings
        else if (PyBytes_Check(x[0])) {
            const char* s = PyBytes_AS_STRING(x[0]);
            rational x;
            if (scan_rational(&s,&x)) {
                const char* p;
                for (p = s; *p; p++) {
                    if (!isspace(*p)) {
                        goto bad;
                    }
                }
                return PyRational_FromRational(x);
            }
            bad:
            PyErr_Format(PyExc_ValueError,
                    "invalid rational literal '%s'",s);
            return 0;
        }
    }

    for (i=0; i<size; i++) {
        PyObject* y;
        int eq;
        x[i] = PyTuple_GET_ITEM(args, i);
        n[i] = PyLong_AsLong(x[i]);
        if (error_converting(n[i])) {
            if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Format(PyExc_TypeError,
                        "expected integer %s, got %s",
                        (i ? "denominator" : "numerator"),
                        x[i]->ob_type->tp_name);
            }
            return 0;
        }
        /* Check that we had an exact integer */
        y = PyLong_FromLong(n[i]);
        if (!y) {
            return 0;
        }
        eq = PyObject_RichCompareBool(x[i],y,Py_EQ);
        Py_DECREF(y);
        if (eq<0) {
            return 0;
        }
        if (!eq) {
            PyErr_Format(PyExc_TypeError,
                    "expected integer %s, got %s",
                    (i ? "denominator" : "numerator"),
                    x[i]->ob_type->tp_name);
            return 0;
        }
    }
    r = make_rational_slow(n[0],n[1]);
    if (PyErr_Occurred()) {
        return 0;
    }
    return PyRational_FromRational(r);
}

/*
 * Returns Py_NotImplemented on most conversion failures, or raises an
 * overflow error for too long ints
 */
#define AS_RATIONAL(dst,object) \
    { \
        dst.n = 0; \
        if (PyRational_Check(object)) { \
            dst = ((PyRational*)object)->r; \
        } \
        else { \
            PyObject* y_; \
            int eq_; \
            long n_ = PyLong_AsLong(object); \
            if (error_converting(n_)) { \
                if (PyErr_ExceptionMatches(PyExc_TypeError)) { \
                    PyErr_Clear(); \
                    Py_INCREF(Py_NotImplemented); \
                    return Py_NotImplemented; \
                } \
                return 0; \
            } \
            y_ = PyLong_FromLong(n_); \
            if (!y_) { \
                return 0; \
            } \
            eq_ = PyObject_RichCompareBool(object,y_,Py_EQ); \
            Py_DECREF(y_); \
            if (eq_<0) { \
                return 0; \
            } \
            if (!eq_) { \
                Py_INCREF(Py_NotImplemented); \
                return Py_NotImplemented; \
            } \
            dst = make_rational_int(n_); \
        } \
    }

static PyObject*
pyrational_richcompare(PyObject* a, PyObject* b, int op) {
    rational x, y;
    int result = 0;
    AS_RATIONAL(x,a);
    AS_RATIONAL(y,b);
    #define OP(py,op) case py: result = rational_##op(x,y); break;
    switch (op) {
        OP(Py_LT,lt)
        OP(Py_LE,le)
        OP(Py_EQ,eq)
        OP(Py_NE,ne)
        OP(Py_GT,gt)
        OP(Py_GE,ge)
    };
    #undef OP
    return PyBool_FromLong(result);
}

static PyObject*
pyrational_repr(PyObject* self) {
    rational x = ((PyRational*)self)->r;
    if (d(x)!=1) {
        return PyUnicode_FromFormat(
                "rational(%ld,%ld)",(long)x.n,(long)d(x));
    }
    else {
        return PyUnicode_FromFormat(
                "rational(%ld)",(long)x.n);
    }
}

static PyObject*
pyrational_str(PyObject* self) {
    rational x = ((PyRational*)self)->r;
    if (d(x)!=1) {
        return PyUnicode_FromFormat(
                "%ld/%ld",(long)x.n,(long)d(x));
    }
    else {
        return PyUnicode_FromFormat(
                "%ld",(long)x.n);
    }
}

static npy_hash_t
pyrational_hash(PyObject* self) {
    rational x = ((PyRational*)self)->r;
    /* Use a fairly weak hash as Python expects */
    long h = 131071*x.n+524287*x.dmm;
    /* Never return the special error value -1 */
    return h==-1?2:h;
}

#define RATIONAL_BINOP_2(name,exp) \
    static PyObject* \
    pyrational_##name(PyObject* a, PyObject* b) { \
        rational x, y, z; \
        AS_RATIONAL(x,a); \
        AS_RATIONAL(y,b); \
        z = exp; \
        if (PyErr_Occurred()) { \
            return 0; \
        } \
        return PyRational_FromRational(z); \
    }
#define RATIONAL_BINOP(name) RATIONAL_BINOP_2(name,rational_##name(x,y))
RATIONAL_BINOP(add)
RATIONAL_BINOP(subtract)
RATIONAL_BINOP(multiply)
RATIONAL_BINOP(divide)
RATIONAL_BINOP(remainder)
RATIONAL_BINOP_2(floor_divide,
    make_rational_int(rational_floor(rational_divide(x,y))))

#define RATIONAL_UNOP(name,type,exp,convert) \
    static PyObject* \
    pyrational_##name(PyObject* self) { \
        rational x = ((PyRational*)self)->r; \
        type y = exp; \
        if (PyErr_Occurred()) { \
            return 0; \
        } \
        return convert(y); \
    }
RATIONAL_UNOP(negative,rational,rational_negative(x),PyRational_FromRational)
RATIONAL_UNOP(absolute,rational,rational_abs(x),PyRational_FromRational)
RATIONAL_UNOP(int,long,rational_int(x),PyLong_FromLong)
RATIONAL_UNOP(float,double,rational_double(x),PyFloat_FromDouble)

static PyObject*
pyrational_positive(PyObject* self) {
    Py_INCREF(self);
    return self;
}

static int
pyrational_nonzero(PyObject* self) {
    rational x = ((PyRational*)self)->r;
    return rational_nonzero(x);
}

static PyNumberMethods pyrational_as_number = {
    pyrational_add,          /* nb_add */
    pyrational_subtract,     /* nb_subtract */
    pyrational_multiply,     /* nb_multiply */
    pyrational_remainder,    /* nb_remainder */
    0,                       /* nb_divmod */
    0,                       /* nb_power */
    pyrational_negative,     /* nb_negative */
    pyrational_positive,     /* nb_positive */
    pyrational_absolute,     /* nb_absolute */
    pyrational_nonzero,      /* nb_nonzero */
    0,                       /* nb_invert */
    0,                       /* nb_lshift */
    0,                       /* nb_rshift */
    0,                       /* nb_and */
    0,                       /* nb_xor */
    0,                       /* nb_or */
    pyrational_int,          /* nb_int */
    0,                       /* reserved */
    pyrational_float,        /* nb_float */

    0,                       /* nb_inplace_add */
    0,                       /* nb_inplace_subtract */
    0,                       /* nb_inplace_multiply */
    0,                       /* nb_inplace_remainder */
    0,                       /* nb_inplace_power */
    0,                       /* nb_inplace_lshift */
    0,                       /* nb_inplace_rshift */
    0,                       /* nb_inplace_and */
    0,                       /* nb_inplace_xor */
    0,                       /* nb_inplace_or */

    pyrational_floor_divide, /* nb_floor_divide */
    pyrational_divide,       /* nb_true_divide */
    0,                       /* nb_inplace_floor_divide */
    0,                       /* nb_inplace_true_divide */
    0,                       /* nb_index */
};

static PyObject*
pyrational_n(PyObject* self, void* closure) {
    return PyLong_FromLong(((PyRational*)self)->r.n);
}

static PyObject*
pyrational_d(PyObject* self, void* closure) {
    return PyLong_FromLong(d(((PyRational*)self)->r));
}

static PyGetSetDef pyrational_getset[] = {
    {(char*)"n",pyrational_n,0,(char*)"numerator",0},
    {(char*)"d",pyrational_d,0,(char*)"denominator",0},
    {0} /* sentinel */
};

static PyTypeObject PyRational_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "experimental_user_dtypes.rational",      /* tp_name */
    sizeof(PyRational),                       /* tp_basicsize */
    0,                                        /* tp_itemsize */
    0,                                        /* tp_dealloc */
    0,                                        /* tp_print */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_reserved */
    pyrational_repr,                          /* tp_repr */
    &pyrational_as_number,                    /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    pyrational_hash,                          /* tp_hash */
    0,                                        /* tp_call */
    pyrational_str,                           /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    "Fixed precision rational numbers",       /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    pyrational_richcompare,                   /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    0,                                        /* tp_methods */
    0,                                        /* tp_members */
    pyrational_getset,                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    0,                                        /* tp_init */
    0,                                        /* tp_alloc */
    pyrational_new,                           /* tp_new */
    0,                                        /* tp_free */
    0,                                        /* tp_is_gc */
    0,                                        /* tp_bases */
    0,                                        /* tp_mro */
    0,                                        /* tp_cache */
    0,                                        /* tp_subclasses */
    0,                                        /* tp_weaklist */
    0,                                        /* tp_del */
    0,                                        /* tp_version_tag */
};

/* NumPy support */

static PyObject*
npyrational_getitem(void* data, void* arr) {
    rational r;
    memcpy(&r,data,sizeof(rational));
    return PyRational_FromRational(r);
}

static int
npyrational_setitem(PyObject* item, void* data, void* arr) {
    rational r;
    if (PyRational_Check(item)) {
        r = ((PyRational*)item)->r;
    }
    else {
        long long n = PyLong_AsLongLong(item);
        PyObject* y;
        int eq;
        if (error_converting(n)) {
            return -1;
        }
        y = PyLong_FromLongLong(n);
        if (!y) {
            return -1;
        }
        eq = PyObject_RichCompareBool(item, y, Py_EQ);
        Py_DECREF(y);
        if (eq<0) {
            return -1;
        }
        if (!eq) {
            PyErr_Format(PyExc_TypeError,
                    "expected rational, got %s", item->ob_type->tp_name);
            return -1;
        }
        r = make_rational_int(n);
    }
    memcpy(data, &r, sizeof(rational));
    return 0;
}




PyMethodDef module_methods[] = {
    {0} /* sentinel */
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "rational",
    NULL,
    -1,
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_rational(void) {
    PyObject *m = NULL;
    PyObject* numpy_str;
    PyObject* numpy;
    int npy_rational;

    /* Specify current dtype version here. Mismatches will be reported */
    int experimental_dtype_version = 0;
    if (import_experimental_dtype_api(experimental_dtype_version) < 0) {
        return NULL;
    }

    import_array();
    if (PyErr_Occurred()) {
        goto fail;
    }
    import_umath();
    if (PyErr_Occurred()) {
        goto fail;
    }
    numpy_str = PyUnicode_FromString("numpy");
    if (!numpy_str) {
        goto fail;
    }
    numpy = PyImport_Import(numpy_str);
    Py_DECREF(numpy_str);
    if (!numpy) {
        goto fail;
    }

    /* Can't set this until we import numpy */
    PyRational_Type.tp_base = &PyArrayDescr_Type;

    /* Initialize rational type object */
    if (PyType_Ready(&PyRational_Type) < 0) {
        goto fail;
    }

    /* Create module */
    m = PyModule_Create(&moduledef);

    if (!m) {
        goto fail;
    }

    /* Add rational type */
    Py_INCREF(&PyRational_Type);
    PyModule_AddObject(m,"rational",(PyObject*)&PyRational_Type);

    return m;

fail:
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "cannot load rational module.");
    }
    if (m) {
        Py_DECREF(m);
        m = NULL;
    }
    return m;
}
