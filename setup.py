from distutils.core import setup, Extension
import subprocess

subprocess.check_call(["scons"])

sources = ['src/cpp/fceumodule.cpp']
libraries = ['fceux', 'SDL']
library_dirs =  ['/usr/local/lib', '/usr/lib', 'src/cpp/fceux']

module1 = Extension('agalt.nes.fceu',
        define_macros = [('PSS_STYLE', '1')],
        sources = sources,
        libraries = libraries,
        library_dirs = library_dirs)

setup(name = 'Agalt',
        version = '1.0',
        description = 'Module for training agents to play video games',
        packages = ['agalt', 'agalt/nes'],
        package_dir = {'': 'src/python'},
        ext_modules = [module1])
