from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="causal-inference-ml",
    version="1.0.0",
    author="wzx11223344",
    author_email="3521257027@QQ.com",
    description="前沿因果推断与机器学习融合引擎 — Double/Debiased ML + Causal Forest + Meta-Learners",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wzx11223344/causal-inference-ml",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
    ],
    keywords="causal-inference machine-learning econometrics double-ml treatment-effects",
)
