from setuptools import setup

setup(
    name='slide_prepper',
    version='0.1.0',    
    description='Preprocessing tools for bright-field Whole Slide Images (WSIs), such as H&E-stained slides',
    url='https://github.com/shuds13/pyexample',
    author='Angela Crabtree',
    author_email='angela.crabtree.88@gmail.com',
    license='None',
    packages=['slide_prepper'],
    install_requires=['opencv-python',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: None',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
    ],
)
