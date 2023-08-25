# -*- coding: utf-8 -*-

import re
from pathlib import Path

from setuptools import (
    find_packages,
    setup,
)

__ascii_art__ = """\n\n \u001b[\u001b[38;5;39m
                                         @.
                                        &  @
                                        @  ,
                                        (
                                                       *
                                            &            @
                                       #    @        @
                                       @             .    ,
                                       *    .             @
                                                     @
                                                     ,    &
                                      (     #             @           @
                                      *     @                       @   @
                                      *     &       /
                                            .       @      #       @     @          *
*   @  %       *       @       &     @                     %                      @    &          *    @     &    @     @
                                                    *      *              @      @      @     @
                                             &                    @                        %
                                                                 .&        @   @
                                                   .        @                &
                                             @                   @
                                                   @
                                             *               @  @
                                                   .            &
                                                              %&
                                              *
                                              .
                                              @    @
                                              
                                               @  .
                                               /
                                                 @
\u001b[0m"""

def find_version(path, varname="__version__"):
    """Parse the version metadata variable in the given file.
    """
    with open(path, 'r') as fobj:
        version_file = fobj.read()
    version_match = re.search(
        r"^{0} = ['\"]([^'\"]*)['\"]".format(varname),
        version_file,
        re.M,
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

# Get the long description from the relevant file
HERE = Path(__file__).parent
with open(HERE / "pypi_description.rst", encoding='utf-8') as f:
    long_description = f.read()

with open(HERE / "requirements.txt") as requires_file:
    requirements = requires_file.read().split("\n")

setup(
    # metadata
    name="bayRing",
    version=find_version(HERE / "bayRing" / "__init__.py"),
    author='Gregorio Carullo, Marina De Amicis, Jaime Redondo-Yuste',
    author_email='gregorio.carullo@ligo.org',
    # contents
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "bayRing = bayRing.bayRing:main",
        ],
    },
    classifiers=[
                 'License :: OSI Approved :: MIT License',
                 'Operating System :: OS Independent',
                 'Development Status :: 4 - Beta',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
    ],
    description='bayRing: Bayesian modelling of Ringdown waveforms with Nested Sampling.',
    license='MIT',
    long_description=long_description,
    url='https://github.com/GCArullo/bayRing',
    project_urls={
      'Bug Tracker': 'https://github.com/GCArullo/bayRing/issues',
      'Source Code': 'https://github.com/GCArullo/bayRing',
    },
    # requirements
    python_requires='>=3',
    install_requires=requirements,
)

try:
    import art
    my_art = art.text2art("            Installed     bayRing") # Return ASCII text (default font)
    print("\u001b[\u001b[38;5;39m{}\u001b[0m".format(my_art))
except: print("* Warning: The `art` package could not be imported. Please consider installing it locally for best visual renditions. The cause of this not being taken care of automatically by the `bayRing` package is that the `art` package is not deployed on conda, hence the conda-build fails and `art` cannot be listed as a requirement for the `bayRing` package.")

print(__ascii_art__)