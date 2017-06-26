from distutils.core import setup, Extension
import glob

sources = ['fceuxmodule.cpp']
libraries = ['fceux', 'SDL']
library_dirs =  ['/usr/local/lib', '/usr/lib']

module1 = Extension('fceux',
        sources = sources,
        libraries = libraries,
        library_dirs = library_dirs)

setup(name = 'FceuxModule',
        version = '1.0',
        description = 'Module for emulating NES games',
        ext_modules = [module1])
