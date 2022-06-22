# flake8: noqa
# pylint: skip-file

from setuptools import setup, find_packages

__version__ = None
exec(open("methods/version_.py").read())

setup(
    name="classificiation-methods",
    python_requires=">=3.7",
    author="Sarah Cameron",
    author_email="sarah.c.griggs92@gmail.com",
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy==1.22.0",
        "pandas==0.25.1",
        "pytest==5.2.1",
    ],
    zip_safe=False,
    url="https://github.com/cameronse68/classification-methods",
    description="Code related to various classification methods",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Natural Language :: English",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3.7",
    ],
)
