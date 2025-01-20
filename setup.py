from setuptools import setup, find_packages

setup(
    name="lego_bricks_ml_vision",
    version="0.1.31",
    packages=find_packages(include=["lego_bricks_ml_vision", "lego_bricks_ml_vision.*", "scripts", "scripts.*"]),
    install_requires=[
        "torch>=1.8.0",
        "ultralytics",
        "pillow",
        "matplotlib",
        "kaggle",
    ],
    entry_points={
        "console_scripts": [
            "run-pipeline=scripts.pipeline:main",
            "run-visualize=scripts.visualize_presentation:main",
        ],
    },
    author="Miguel Di Lalla",
    author_email="tu_email@example.com",
    description="A package for LEGO brick detection and visualization using YOLO.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MiguelDiLalla/LEGO_Bricks_ML_Vision",
    license="Apache License 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
    ],
    python_requires=">=3.8",
)
