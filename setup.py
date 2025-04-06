from setuptools import setup, find_packages

setup(
    name='taylorgrad',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=2.2.2',
        'setuptools==72.1.0',
        'pytest==7.4.4',
    ],
    author='Alex Chadyuk',
    description='Adaptive learning rate for stochastic gradient descent',
    url='https://github.com/chadyuk/taylorgrad',
    license='MIT',
    author_email='alex@chadyuk.com',
)