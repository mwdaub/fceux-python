from distutils.core import setup, Extension

module1 = Extension('fceux', sources = ['fceuxmodule.c'])

setup(name = 'FceuxModule',
        version = '1.0',
        description = 'Module for emulating NES games',
        ext_modules = [module1])
