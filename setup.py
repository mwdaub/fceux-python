from distutils.core import setup, Extension
import subprocess

subprocess.check_call(["scons"])

sources = ['src/fceuxmodule.cpp']
libraries = ['fceux', 'SDL']
library_dirs =  ['/usr/local/lib', '/usr/lib', 'src/fceux']

module1 = Extension('fceux',
        define_macros = [('PSS_STYLE', '1')],
        sources = sources,
        libraries = libraries,
        library_dirs = library_dirs)

setup(name = 'FceuxModule',
        version = '1.0',
        description = 'Module for emulating NES games',
        ext_modules = [module1])
