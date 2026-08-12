"""Build the native pathFinder extension on Windows and Linux."""

import os

from setuptools import Extension, setup


# MSVC uses slash-prefixed compiler options, while GCC and Clang use
# dash-prefixed options. MSVC's first explicit standard mode is C++14, which
# supports all C++11 features used by this extension.
extra_compile_args = ["/std:c++14"] if os.name == "nt" else ["-std=c++11"]
# A Python extension DLL does not require an application manifest. Disabling
# manifest embedding also avoids relying on rc.exe being discoverable from
# stripped-down shells such as VS Code's integrated terminal.
extra_link_args = ["/MANIFEST:NO"] if os.name == "nt" else []

setup(
    name="pathFinder",
    version="1.0",
    ext_modules=[
        Extension(
            "pathFinder",
            sources=["pathFinder.cpp"],
            language="c++",
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
)
