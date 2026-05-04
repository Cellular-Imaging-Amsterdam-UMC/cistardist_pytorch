#include <Python.h>
#include <math.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "numpy/arrayobject.h"
#include "clipper.hpp"
#include "nanoflann.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
struct PointCloud2D {
    struct Point {
        T x, y;
    };

    std::vector<Point> pts;

    inline size_t kdtree_get_point_count() const { return pts.size(); }

    inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
        return dim == 0 ? pts[idx].x : pts[idx].y;
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const {
        return false;
    }
};

static inline float area_from_path(const ClipperLib::Path& path) {
    float area = 0.0f;
    const int n = static_cast<int>(path.size());
    for (int i = 0; i < n; i++) {
        area += path[i].X * path[(i + 1) % n].Y - path[i].Y * path[(i + 1) % n].X;
    }
    return 0.5f * std::abs(area);
}

static inline bool bbox_intersect(
    const float bbox_a_x1,
    const float bbox_a_x2,
    const float bbox_a_y1,
    const float bbox_a_y2,
    const float bbox_b_x1,
    const float bbox_b_x2,
    const float bbox_b_y1,
    const float bbox_b_y2
) {
    return bbox_b_x1 <= bbox_a_x2 && bbox_a_x1 <= bbox_b_x2 &&
           bbox_b_y1 <= bbox_a_y2 && bbox_a_y1 <= bbox_b_y2;
}

static inline float poly_intersection_area(
    const ClipperLib::Path& poly_a_path,
    const ClipperLib::Path& poly_b_path
) {
    ClipperLib::Clipper clipper;
    ClipperLib::Paths result;

    clipper.AddPath(poly_a_path, ClipperLib::ptClip, true);
    clipper.AddPath(poly_b_path, ClipperLib::ptSubject, true);
    clipper.Execute(ClipperLib::ctIntersection, result, ClipperLib::pftNonZero, ClipperLib::pftNonZero);

    float area_inter = 0.0f;
    for (size_t i = 0; i < result.size(); i++) {
        area_inter += area_from_path(result[i]);
    }
    return area_inter;
}

static void free_nms_buffers(
    float* areas,
    bool* suppressed,
    ClipperLib::Path* poly_paths,
    float* bbox_x1,
    float* bbox_x2,
    float* bbox_y1,
    float* bbox_y2,
    float* radius_outer
) {
    delete[] areas;
    delete[] suppressed;
    delete[] poly_paths;
    delete[] bbox_x1;
    delete[] bbox_x2;
    delete[] bbox_y1;
    delete[] bbox_y2;
    delete[] radius_outer;
}

static PyObject* c_non_max_suppression_inds(PyObject*, PyObject* args) {
    PyArrayObject* dist = NULL;
    PyArrayObject* points_arr = NULL;
    float threshold;
    int verbose;
    int use_kdtree;
    int use_bbox;

    if (!PyArg_ParseTuple(
            args,
            "O!O!iiif",
            &PyArray_Type,
            &dist,
            &PyArray_Type,
            &points_arr,
            &use_kdtree,
            &use_bbox,
            &verbose,
            &threshold
        )) {
        return NULL;
    }

    if (PyArray_NDIM(dist) != 2 || PyArray_TYPE(dist) != NPY_FLOAT32 ||
        PyArray_NDIM(points_arr) != 2 || PyArray_TYPE(points_arr) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_ValueError, "dist and points must be contiguous float32 arrays.");
        return NULL;
    }

    npy_intp* dims = PyArray_DIMS(dist);
    const int n_polys = static_cast<int>(dims[0]);
    const int n_rays = static_cast<int>(dims[1]);

    npy_intp* point_dims = PyArray_DIMS(points_arr);
    if (point_dims[0] != n_polys || point_dims[1] != 2) {
        PyErr_SetString(PyExc_ValueError, "points must have shape (n_polys, 2).");
        return NULL;
    }

    float* bbox_x1 = new float[n_polys];
    float* bbox_x2 = new float[n_polys];
    float* bbox_y1 = new float[n_polys];
    float* bbox_y2 = new float[n_polys];
    float* radius_outer = new float[n_polys];
    float* areas = new float[n_polys];
    bool* suppressed = new bool[n_polys];
    ClipperLib::Path* poly_paths = new ClipperLib::Path[n_polys];

    const float* const points = static_cast<float*>(PyArray_DATA(points_arr));
    const float angle_step = static_cast<float>(2.0 * M_PI / n_rays);
    int count_suppressed = 0;

#pragma omp parallel for
    for (int i = 0; i < n_polys; i++) {
        suppressed[i] = false;
    }

    if (verbose) {
        std::printf("Non Maximum Suppression (2D) ++++\n");
        std::printf(
            "NMS: n_polys = %d\nNMS: n_rays = %d\nNMS: thresh = %.3f\nNMS: use_bbox = %d\nNMS: use_kdtree = %d\n",
            n_polys,
            n_rays,
            threshold,
            use_bbox,
            use_kdtree
        );
#ifdef _OPENMP
        std::printf("NMS: using OpenMP with %d thread(s)\n", omp_get_max_threads());
#endif
        std::fflush(stdout);
    }

    for (int i = 0; i < n_polys; i++) {
        ClipperLib::Path clip;
        const float py = points[2 * i];
        const float px = points[2 * i + 1];
        float max_radius_outer = 0.0f;

        for (int k = 0; k < n_rays; k++) {
            const float d = *static_cast<float*>(PyArray_GETPTR2(dist, i, k));
            const float y = py + d * std::sin(angle_step * k);
            const float x = px + d * std::cos(angle_step * k);

            if (k == 0) {
                bbox_x1[i] = x;
                bbox_x2[i] = x;
                bbox_y1[i] = y;
                bbox_y2[i] = y;
            } else {
                bbox_x1[i] = std::min(x, bbox_x1[i]);
                bbox_x2[i] = std::max(x, bbox_x2[i]);
                bbox_y1[i] = std::min(y, bbox_y1[i]);
                bbox_y2[i] = std::max(y, bbox_y2[i]);
            }

            clip << ClipperLib::IntPoint(
                static_cast<ClipperLib::cInt>(std::llround(x)),
                static_cast<ClipperLib::cInt>(std::llround(y))
            );
            max_radius_outer = std::max(d, max_radius_outer);
        }

        radius_outer[i] = max_radius_outer;
        poly_paths[i] = clip;
        areas[i] = area_from_path(clip);
    }

    PointCloud2D<float> cloud;
    cloud.pts.resize(n_polys);
    float max_dist = 0.0f;
    for (int i = 0; i < n_polys; i++) {
        cloud.pts[i].x = points[2 * i];
        cloud.pts[i].y = points[2 * i + 1];
        max_dist = std::max(radius_outer[i], max_dist);
    }

    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloud2D<float>>,
        PointCloud2D<float>,
        2>
        kd_tree_t;

    kd_tree_t index(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    nanoflann::SearchParams params;
    std::vector<std::pair<size_t, float> > results;

    if (use_kdtree) {
        index.buildIndex();
    }

    for (int i = 0; i < n_polys - 1; i++) {
        if (suppressed[i]) {
            continue;
        }

        if (PyErr_CheckSignals() == -1) {
            free_nms_buffers(areas, suppressed, poly_paths, bbox_x1, bbox_x2, bbox_y1, bbox_y2, radius_outer);
            return NULL;
        }

        if (use_kdtree) {
            const float radius = max_dist + radius_outer[i];
            index.radiusSearch(&points[2 * i], radius * radius, results, params);
        } else {
            results.resize(n_polys - i);
            for (size_t n = 0; n < results.size(); n++) {
                results[n].first = i + n;
            }
        }

#ifdef __APPLE__
#pragma omp parallel for reduction(+ : count_suppressed) shared(suppressed)
#else
#pragma omp parallel for schedule(dynamic) reduction(+ : count_suppressed) shared(suppressed)
#endif
        for (long neigh = 0; neigh < static_cast<long>(results.size()); neigh++) {
            const long j = static_cast<long>(results[neigh].first);

            if (suppressed[j] || j <= i) {
                continue;
            }

            if (use_bbox &&
                !bbox_intersect(
                    bbox_x1[i],
                    bbox_x2[i],
                    bbox_y1[i],
                    bbox_y2[i],
                    bbox_x1[j],
                    bbox_x2[j],
                    bbox_y1[j],
                    bbox_y2[j]
                )) {
                continue;
            }

            const float area_inter = poly_intersection_area(poly_paths[i], poly_paths[j]);
            const float overlap = area_inter / std::min(areas[i] + 1.e-10f, areas[j] + 1.e-10f);
            if (overlap > threshold) {
                count_suppressed += 1;
                suppressed[j] = true;
            }
        }
    }

    npy_intp dims_result[1];
    dims_result[0] = n_polys;
    PyArrayObject* result = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, dims_result, NPY_BOOL));

    for (int i = 0; i < n_polys; i++) {
        *static_cast<bool*>(PyArray_GETPTR1(result, i)) = !suppressed[i];
    }

    if (verbose) {
        std::printf("NMS: Suppressed polygons: %d / %d\n", count_suppressed, n_polys);
        std::fflush(stdout);
    }

    free_nms_buffers(areas, suppressed, poly_paths, bbox_x1, bbox_x2, bbox_y1, bbox_y2, radius_outer);

    return PyArray_Return(result);
}

static PyMethodDef methods[] = {
    {"c_non_max_suppression_inds", c_non_max_suppression_inds, METH_VARARGS, "2D StarDist polygon NMS"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_c_nms",
    NULL,
    -1,
    methods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit__c_nms(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
