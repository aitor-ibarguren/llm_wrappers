from setuptools import setup, find_packages

setup(
    name='llm_wrappers',
    version='0.1',
    packages=find_packages(),
    description='LLM wrapper classes in Python for a simple development of LLM-based applications',
    author='Aitor Ibarguren',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
