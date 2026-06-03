from setuptools import setup, find_packages

setup(
    name="trajectory-prediction-attention",
    version="0.1.0",
    description="Multi-Level Attention Trajectory Prediction (MGFNet) for Argoverse 1",
    author="ML Researcher",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
    ],
)
