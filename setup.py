from setuptools import setup

setup(name='masseff', 
      version='0.1',
      description='Effective mass calculator',
      author='wei.chen',
      author_email='wch3n@pm.me',
      license='MIT',
      packages=['masseff'],
      install_requires=[
        'numpy', 'pymatgen'
      ],
      zip_safe=False)
