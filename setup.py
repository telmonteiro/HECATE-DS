from setuptools import setup, find_packages

setup(name = 'HECATE',
    version = "0.1.0",
    description = 'HarvEsting loCAl specTra with Exoplanets - Doppler Shadow',
    url = 'https://github.com/telmonteiro/HECATE-DS/',
    license = 'MIT',
    author = 'Telmo Monteiro',
    author_email = 'telmo.monteiro@astro.up.pt',
    keywords = ['astronomy'],
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['*.csv'],},
    install_requires = ['numpy', 'dynesty', 'matplotlib', 'scipy']
)