from setuptools import setup, find_packages


install_requires = [
  'opencv-python',
  'typing',
  'tqdm',
  'Pillow',
  'torch',
  'transformers',
  'diffusers',
  'accelerate',
]

setup(
  name='viddiffusion',
  version='0.1',
  install_requires=install_requires,
  packages=find_packages()
)