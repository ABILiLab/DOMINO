from setuptools import find_packages, setup

__lib_name__ = "DOMINO"
__lib_version__ = "0.1.0"
__description__ = "Identifying distinct spatial domains from clear cell and endometrioid ovarian carcinoma using DOMINO"
__url__ = "https://github.com/ABILiLab/DOMINO"
__author__ = ""
__author_email__ = ""
__license__ = "MIT"
# 
__requires__ = []
__long_description__ = open('README.md').read()

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ["DOMINO"],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
    long_description = __long_description__,
    long_description_content_type="text/markdown"
)
