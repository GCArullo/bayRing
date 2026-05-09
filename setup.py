# -*- coding: utf-8 -*-

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

# Get the long description from the relevant file
HERE = Path(__file__).parent
with open(HERE / "pypi_description.rst", encoding='utf-8') as f:
    long_description = f.read()

setup(
    # metadata
    name="bayRing",
    use_scm_version=True,
    # contents
    packages=find_packages(),
    long_description=long_description,
)

try:
    import art
    my_art = art.text2art("            Installed     bayRing") # Return ASCII text (default font)
    print("\u001b[\u001b[38;5;39m{}\u001b[0m".format(my_art))
except: print("* Warning: The `art` package could not be imported. Please consider installing it locally for best visual renditions. The cause of this not being taken care of automatically by the `bayRing` package is that the `art` package is not deployed on conda, hence the conda-build fails and `art` cannot be listed as a requirement for the `bayRing` package.")

print(__ascii_art__)
