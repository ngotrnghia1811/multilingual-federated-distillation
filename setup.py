import setuptools

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="multilingual-federated-distillation",
    version="1.0.0",
    author="Nghia Trung Ngo",
    author_email="nghian@uoregon.edu",
    description="Multi-lingual Federated Distillation (MlFD) for efficient FL over multilingual NLP models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nghiatrngo/MlFD",
    packages=setuptools.find_packages(exclude=["tests", "data"]),
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="federated-learning, multilingual, knowledge-distillation, nlp",
)
