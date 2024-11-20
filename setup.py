from setuptools import setup

setup(
   name='spinr',
   version='0.0',
   description='Spiking Neural Resonator Network',
   author='Nico Reeb',
   author_email='nico.reeb@tum.de',
   packages=['spinr'],  #same as name
   install_requires=['numpy', 'matplotlib', 'scipy', 'cupy', 'pycuda', 'pandas'], #external packages as dependencies
)