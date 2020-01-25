from setuptools import setup, find_packages

setup(
    name='spoint',
    version='0.1',
    license='MIT',
    packages=find_packages(exclude=['tests', 'notebooks']),
    description='SuperPoint PyTorch implementation',
    url='https://github.com/BillMills/python-package-example',
    author='Alexander Tanchenko',
    author_email='aletan@protonmail.com',
    install_requires=[
        'torch >= 1.4',
        'numpy',
        'scipy',
        'tqdm',
        'pyyaml',
        'opencv-python >= 3.4.2'
    ],
    extras_require={
        'develop': [
            'flake8',
            'jupyter',
            'matplotlib'
        ]
    }
)
