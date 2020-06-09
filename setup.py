import setuptools

with open("README.md", "r")as fh:
    des = fh.read()

setuptools.setup(
    name="diamond-pkg-PeterA182",
    version="0.0.1",
    author="Peter Altamura",
    description="A class structure for manipulating Normalized MySportsFeed API baseball data",
    url="https://github.com/PeterA182/diamond",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS"
    ],
    python_requires=">=3.6"
)

