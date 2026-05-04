# Notices

This project implements StarDist-compatible 2D inference behavior for PyTorch.

The polygon geometry and non-maximum suppression behavior in
`cistardist_pytorch.geometry` and `cistardist_pytorch.nms` is adapted from the
BSD-3-Clause licensed StarDist project:

https://github.com/stardist/stardist

StarDist citation:

Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers. Cell Detection
with Star-convex Polygons. MICCAI 2018.

The compiled NMS extension vendors the following third-party components from
StarDist's source distribution:

- Clipper 6.4.2 by Angus Johnson, distributed under the Boost Software License
  1.0. See `cistardist_pytorch/lib/external/clipper/LICENSE.txt`.
- nanoflann, distributed under the BSD license. See
  `cistardist_pytorch/lib/external/nanoflann/LICENSE.txt`.
