from setuptools import setup, find_packages

setup(
    name='federated-learning-framework',
    version='0.0.1',
    description='A modular and extensible framework for federated learning applications.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Mehrdad Javadi',
    author_email='mehrdaddjavadi@gmail.com',
    url='https://github.com/mehrdaddjavadi/federated_learning_framework',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: Free for academic use',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='federated learning, machine learning, deep learning, encryption',
    install_requires=[
        'numpy',
        'tensorflow',
        'websockets',
        'pytest',
        'tenseal'
    ],
    python_requires='>=3.6',
    include_package_data=True,
    zip_safe=False,
)
