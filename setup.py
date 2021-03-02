import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dataset_iterator",
    version="0.2.1",
    author="Jean Ollion",
    author_email="jean.ollion@polytechnique.org",
    description="data iterator for images contained in dataset files such as hdf5 or PIL readable files. Images can be contained in several files. Based on tensorflow.keras.preprocessing.image.Iterator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jeanollion/dataset_iterator.git",
    download_url = 'https://github.com/jeanollion/dataset_iterator/archive/v_01.tar.gz',
    keywords = ['Iterator', 'Dataset', 'Image', 'Numpy'],
    packages=setuptools.find_packages(),
    classifiers=[ #https://pypi.org/classifiers/
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3',
    install_requires=['h5py>=2.9', 'numpy', 'scipy', 'scikit-learn', 'tensorflow']
)
