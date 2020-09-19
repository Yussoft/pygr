import setuptools

long_description = ""
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pygr",
    version="0.0.1",
    author="Luvo, Alberto y Yus.",
    author_email="yussdc@gmail.com",
    description="A repo to try to do something cool with ML.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Yussoft/pygr",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
