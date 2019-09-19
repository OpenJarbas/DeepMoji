from setuptools import setup

setup(
    name='simple_deepmoji',
    version='0.1',
    packages=['simple_deepmoji'],
    description='DeepMoji library',
    include_package_data=True,
    install_requires=[
        'emoji==0.4.5',
        'h5py==2.7.0',
        'Keras==2.0.9',
        'numpy==1.13.1',
        'scikit-learn==0.19.0',
        'text-unidecode==1.0',
    ],
)
