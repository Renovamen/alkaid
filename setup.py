from os import path
from setuptools import setup, find_packages

current_path = path.abspath(path.dirname(__file__))

# load content from `README.md`
def readme():
    readme_path = path.join(current_path, 'README.md')
    with open(readme_path, encoding = 'utf-8') as fp:
        return fp.read()

setup(
    name = 'metallic',
    version = "0.0.1",
    packages = find_packages(),
    description = 'Reinforcement learning.',
    long_description = readme(),
    long_description_content_type = 'text/markdown',
    keywords=['pytorch', 'machine learning', 'reinforcement learning'],
    license = 'MIT',
    author = 'Xiaohan Zou',
    author_email = 'renovamenzxh@gmail.com',
    url = 'https://github.com/Renovamen/alkaid',
    include_package_data = True,
    install_requires = [
        'numpy!=1.16.0,<1.20.0',
        'torch>=1.4.0',
        'box2d-py>=2.3.8',
        'gym>=0.18.0',
        'matplotlib'
    ]
)
