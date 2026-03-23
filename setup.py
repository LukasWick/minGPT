from setuptools import setup

setup(name='minGPT',
      version='0.0.1',
      author='Andrej Karpathy',
      packages=['mingpt', 'mingpt2'],
      description='A PyTorch re-implementation of GPT',
      license='MIT',
      install_requires=[
            'torch',
      ],
)
