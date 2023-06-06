from setuptools import setup

setup(
    name='InstructZero',
    version='1.0',
    description='',
    author='Lichang Chen',
    author_email='bobchen@umd.edu',
    packages=['InstructZero'],
    install_requires=[
        'numpy',
        'openai',
        'fire',
        'tqdm',
        'gradio',
        'torch',
        'botorch',
        'transformers',
    ],
)
