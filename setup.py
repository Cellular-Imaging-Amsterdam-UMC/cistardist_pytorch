from __future__ import annotations

from numpy import get_include
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExtOpenMP(build_ext):
    openmp_compile_args = {
        "msvc": [["/openmp"]],
        "intel": [["-qopenmp"]],
        "*": [["-fopenmp"], ["-Xpreprocessor", "-fopenmp"]],
    }
    openmp_link_args = {
        "msvc": [[]],
        "intel": [["-qopenmp"]],
        "*": [["-fopenmp"], ["-Xpreprocessor", "-fopenmp"]],
    }

    def build_extension(self, ext: Extension) -> None:
        compiler = self.compiler.compiler_type.lower()
        if compiler.startswith("intel"):
            compiler = "intel"
        if compiler not in self.openmp_compile_args:
            compiler = "*"

        original_compile_args = list(ext.extra_compile_args)
        original_link_args = list(ext.extra_link_args)
        if compiler == "msvc":
            original_compile_args = [arg for arg in original_compile_args if not arg.startswith("-std=")]

        for compile_args, link_args in zip(self.openmp_compile_args[compiler], self.openmp_link_args[compiler]):
            try:
                ext.extra_compile_args = original_compile_args + compile_args
                ext.extra_link_args = original_link_args + link_args
                return super().build_extension(ext)
            except Exception:
                print(f">>> compiling with '{' '.join(compile_args)}' failed")

        print(">>> compiling with OpenMP support failed, re-trying without")
        ext.extra_compile_args = original_compile_args
        ext.extra_link_args = original_link_args
        return super().build_extension(ext)


lib_dir = "cistardist_pytorch/lib"
clipper_dir = f"{lib_dir}/external/clipper"
nanoflann_dir = f"{lib_dir}/external/nanoflann"


setup(
    cmdclass={"build_ext": BuildExtOpenMP},
    ext_modules=[
        Extension(
            "cistardist_pytorch._c_nms",
            sources=[
                f"{lib_dir}/c_nms.cpp",
                f"{clipper_dir}/clipper.cpp",
            ],
            include_dirs=[
                get_include(),
                clipper_dir,
                nanoflann_dir,
            ],
            extra_compile_args=["-std=c++11"],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
    ],
)
