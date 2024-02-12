import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="graphlearning_mvpa",
    version="1.0",
    author="Ari Kahn",
    author_email="arikahn@seas.upenn.edu",
    description="nipype scripts to run mvpa analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'nipype',
        'nilearn',
        'pandas',
    ],
)
