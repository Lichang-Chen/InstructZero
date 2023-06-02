from setuptools import setup

setup(
    name='InstructZero',
    version='1.0',
    description='',
    author='Lichang Chen',
    author_email='bobchen@umd.edu',
    packages=['InstructZero',
              'automatic_prompt_engineer.evaluation'],
    package_data={'automatic_prompt_engineer': ['configs/*']},
    install_requires=[
        'numpy',
        'openai',
        'fire',
        'tqdm',
        'gradio',
        'torch',
        'botorch'
    ],
)
