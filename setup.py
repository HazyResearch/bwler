from setuptools import setup, find_packages

setup(
    name="bwler",
    version="0.1.0",
    description="Barycentric Weight Layer Elucidates a Precision-Conditioning Tradeoff for PINNs",
    author="Jerry Liu, Yasa Baig, Denise Hui Jean Lee, Rajat Vadiraj Dwaraknath, Atri Rudra, Chris Ré",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "einops==0.8.0",
        "jupytext==1.16.5",
        "matplotlib==3.10.0",
        "numpy==2.2.0",
        "scipy==1.15.1",
        "torch==2.3.0",
        "tqdm==4.67.1",
        "wandb==0.19.4",
    ],
) 