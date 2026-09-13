import setuptools

with open('README.md') as f:
    long_description = ''.join(f.readlines())

setuptools.setup(
    name="CVineMarketGen",
    version="0.1.0",
    author="Mamadou Yamar Thioub",
    author_email="mamadou-yamar.thioub@hec.ca",
    description="C-vine copula financial market generator with moment and tail dependence targeting, "
                "and the Fleishman / Vale-Maurelli benchmark. Code of the third article of the thesis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mamadouyamar/CVineMarketGen",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=['cvinemarketgen'],
    install_requires=['numpy', 'scipy', 'pandas', 'matplotlib', 'pyvinecopulib', 'statsmodels', 'requests'],
    python_requires='>=3.8',
)
