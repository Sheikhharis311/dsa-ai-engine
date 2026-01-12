from setuptools import setup

setup(
    name="Haris", # PyPI par unique hona chahiye
    version="0.0.1", # aapke module ke docstring se match
    py_modules=["main"], # kyunki sirf main.py hai
    description="Advanced DSA & AI Engine - single file implementation",
    author="Sheikh haris raza",
    license="MIT",
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

)
