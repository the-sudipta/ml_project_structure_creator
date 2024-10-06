from setuptools import setup, find_packages

setup(
    name='ml_project_structure_creator',
    version='0.2.0',
    packages=find_packages(),  # Automatically find packages
    scripts=['create_ml_structure.py'],  # Your main script
    description='A script to create a machine learning project structure with EDA code.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Sudipta Kumar Das',
    author_email='dip.kumar020@gmail.com',
    url='https://github.com/the-sudipta/ml_project_structure_creator',
    license='GNU General Public License v3 (GPLv3)',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10.11',  # Adjust according to your script requirements
    entry_points={
        'console_scripts': [
            'create_ml_structure=create_ml_structure:main',  # Adjusting for the function in your script
        ],
    },
)
