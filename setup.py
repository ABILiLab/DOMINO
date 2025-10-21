from setuptools import find_packages, setup

__lib_name__ = "domino-spatial"
__lib_version__ = "0.1.3"
__description__ = "Identifying distinct spatial domains from clear cell and endometrioid ovarian carcinoma using DOMINO"
__url__ = "https://github.com/ABILiLab/DOMINO"
__author__ = "Pan Jia, Wenjun Liu, Ning Liu, and Fuyi Li"
__author_email__ = "jiapan@nwafu.edu.cn; nora.liu@adelaide.edu.au; ning.liu@adelaide.edu.au; fuyi.li@nwafu.edu.cn"
__requires__ = []
__long_description__ = open('README.md').read()

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = "MIT",
    packages = ["domino_spatial"],
    entry_points={
        'console_scripts': [
            'domino-spatial=domino_spatial.cli:main',  
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",   
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
    long_description = __long_description__,
    long_description_content_type="text/markdown"
)
