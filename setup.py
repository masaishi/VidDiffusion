from setuptools import setup, find_packages


install_requires = [
  'opencv-python',
  'tqdm',
  'Pillow',
  'torch',
  'transformers',
  'diffusers',
  'accelerate',
]

setup(
  name='viddiffusion',
  author='Masamune Ishihara',
  author_email='masaishi_masa@yahoo.co.jp',
  maintainer='Masamune Ishihara',
  maintainer_email='masaishi_masa@yahoo.co.jp',
  description="VidDiffusion is a Python library that provides vid2vid pipeline by using Hugging Face's `diffusers`.",
  long_description=open('README.md').read(),
  long_description_content_type='text/markdown',
  license='Apache License 2.0',
  url='https://github.com/masaishi/VidDiffusion',
  version=open('viddiffusion/__init__.py').readlines()[-1].split()[-1].strip().strip("'"),
  python_requires='>=3.6',
  install_requires=install_requires,
  packages=find_packages()
)