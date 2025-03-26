#!/usr/bin/env python3
"""
Setup script for Web Visual Similarity Engine.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().strip().split("\n")

setup(
    name="websim",
    version="1.0.0",
    author="Web Similarity Team",
    author_email="info@websimilarity.com",
    description="A tool for finding visually similar web pages",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/websim",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "websim-index=websim.indexer:main",
            "websim-query=websim.query:main",
            "websim-compare=websim.compare:main",
        ],
    },
) 