#! /usr/bin/env python
# Authors: Olivier Grisel <olivier.grisel@ensta.org>
# LICENSE: MIT
from distutils.core import setup

setup(
    name="oglearn",
    version="0.1-dev",
    description="Experimental utilities and extensions for scikit-learn",
    maintainer="Olivier Grisel",
    maintainer_email="olivier.grisel@ensta.org",
    license="MIT",
    url='http://github.com/ogrisel/oglearn',
    packages=[
        'oglearn',
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Development Status :: 3 - Alpha',
    ],
)
