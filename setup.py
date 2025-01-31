from setuptools import setup, find_packages

setup(
    name="lego-vision-cli",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "torch",
        "argparse",
        "pyyaml",
        "shutil",
        "glob2"
    ],
    entry_points={
        "console_scripts": [
            "lego-setup=scripts.pipeline_setup:main",
            "lego-train=scripts.pipeline_train:main",
            "lego-utils=scripts.pipeline_utils:main",
            "lego-export-results=scripts.cli:run_export_results"
        ]
    },
    author="Miguel Di Lalla",
    description="CLI for managing LEGO Bricks ML pipeline",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MiguelDiLalla/LEGO_Bricks_ML",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
