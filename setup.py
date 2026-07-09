import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fr:
    installation_requirements = fr.readlines()

setuptools.setup(
    name="promptflow",
    version="0.0.3",
    author="Goncalo Faria",
    author_email="goncalofaria.research@gmail.com",
    description="A package for building and running workflows.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/goncalorafaria/promptflow",
    packages=setuptools.find_packages(),
    install_requires=installation_requirements,
    python_requires=">=3.8.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
