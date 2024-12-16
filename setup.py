from setuptools import setup, find_packages

setup(
    name="PGM_sujet_3",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["networkx", "torch", "numpy",
                      "matplotlib", "scikit-learn", "scipy"],
)