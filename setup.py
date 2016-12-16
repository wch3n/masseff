from setuptools import setup

setup(name='masseff', 
      version='0.1',
      description='Effective mass calculator',
      author='wayn3',
      author_email='waynechen@gmail.com',
      license='MIT',
      packages=['masseff'],
      install_requires=[
        'numpy', 'pymatgen'
      ],
      zip_safe=False)
