from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy>=1.18.1", "pandas>=1.0.3", "dtw-python==1.1.6", "pycausalimpact==0.0.14"]

setup(
    name="pycausalmatch",
    version="0.0.3",
    author="Rishi Jumani",
    author_email="unbiased.modeler@gmail.com",
    description="Causal Impact of an intervention integrated with control group selection",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/unbiasedmodeler/pycausalmatch",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
    ],
)