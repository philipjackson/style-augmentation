from setuptools import setup, find_packages
setup(
    name='styleaug',
    version='1.0',
    description='Neural style randomization in PyTorch, for data augmentation',
    author='Philip Jackson',
    author_email='p.t.g.jackson@durham.ac.uk',
    url='https://github.com/philipjackson/styleaug',
    license='MIT',
    install_requires=['torch','torchvision','numpy'],
    packages=find_packages(),
    include_package_data=True
)