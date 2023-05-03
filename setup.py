from setuptools import setup, find_packages
import pytorchfi


VERSION=1.0
DESCRIPTION ='pytorchfi description'
LONG_DESCRIPTION='pytorchfi description'


#seting up

setup(
    name="pytorchfi",
    version=VERSION,
    author="Juan David Guerrero",
    author_email="",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url="https://github.com/divadnauj-GB/pytorchfi_SC",
    packages=find_packages(exclude=('tests', 'scripts', 'Demo')),
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.12.1',
        'torchvision>=0.13.1',
        'numpy',
        'pyyaml>=5.4.1',
        'scipy',
        'cython',
        'pycocotools>=2.0.2'
    ],
    extras_require={
        'test': ['pytest']
    },
    keywords=['pytorch','python'],
    classifiers=[
        "Development Status :: Apha",
        "Programming Language :: Python :: 3",
    ]
)
