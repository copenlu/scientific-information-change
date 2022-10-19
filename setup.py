import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name="scientific-information-change",
    version="1.0",
    author="Dustin Wright",
    author_email="dw@di.ku.dk",
    description="This package is used to estimate the information matching score (IMS) of scientific sentences. IMS is a measure of the similarity of the information contained in scientific findings i.e. how similar are the scientific findings described by two scientific sentences. This can be used to match sentences describing the same scientific findings or measure the degree to which sentences differ in the findings they describe.",
    python_requires=">=3.6",
    include_package_data=True,
    packages=['scientific_information_change'],
    install_requires=[
        'sentence-transformers==2.2.2',
        'numpy'
    ],
    long_description=long_description,
    long_description_content_type="text/markdown"
    #package_data = {'':['data']}
)
