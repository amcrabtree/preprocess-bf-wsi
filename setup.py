from setuptools import setup

setup(
    name='slide_prepper',
    version='1.0',    
    description='Preprocessing tools for bright-field Whole Slide Images (WSIs), such as H&E-stained slides',
    url='https://github.com/amcrabtree/preprocess-bf-wsi',
    author='Angela Crabtree',
    author_email='angela.crabtree.88@gmail.com',
    license='None',
    packages=['slide_prepper'],
    install_requires=['opencv-python',
                      'numpy',                     
                      ],
)
