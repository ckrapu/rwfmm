from setuptools import setup

setup(
    name='rwfmm',
    version='0.1',
    description='Code for random-walk functional mixed model',
    author='Christopher Krapu',
    author_email='ckrapu@gmail.com',
    py_modules=["models",'utilities'],
    install_requires=['theano','numpy','pymc3','matplotlib','pandas']
)
