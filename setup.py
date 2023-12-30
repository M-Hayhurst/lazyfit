from setuptools import setup

setup(
    name='lazyfit',
    version='1.0',
    author='Martin Hayhurst Appel',
    url='https://github.com/M-Hayhurst/lazyfit/',
    packages=['lazyfit'],
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'
    ],
    package_dir={'lazyfit': 'lazyfit', },
)
