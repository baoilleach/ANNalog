from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else ""

setup(
    name="annalog",
    version="1.0",  # bump version before publishing
    packages=find_packages(),
    include_package_data=True,
    package_data={"annalog": ["ckpt_and_vocab/*"]},

    description="ANNalog — a SMILES-to-SMILES seq2seq model for medchem analogue generation",
    long_description=README,
    long_description_content_type="text/markdown",

    author="Wei Dai",
    author_email="bty415@qmul.ac.uk",
    license="MIT",
    url="https://github.com/DVNecromancer/ANNalog",

    install_requires=[
        "partialsmiles @ git+https://github.com/baoilleach/partialsmiles.git",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
