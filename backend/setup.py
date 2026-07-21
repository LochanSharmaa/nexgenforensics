from setuptools import find_packages, setup

setup(
    name="nexgen-facial-engine",
    version="0.1.0",
    description="Commercial facial recognition engine for NexGen Identity",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "fastapi==0.115.6",
        "uvicorn[standard]==0.34.0",
        "python-multipart==0.0.20",
        "pillow==11.1.0",
        "pydantic==2.10.5",
        "numpy==2.2.1",
        "cryptography==44.0.0",
    ],
)
