#Responsible in creating my Ml application as a package, and anyone can install the project and use it

from setuptools import find_packages,setup
from typing import List

## we add '-e .' in setup,it automatically run setup.py and so we do not need it run in reqirements
HYPEN_E_DOT="-e ."

def get_requirements(file_path:str)->List[str]:

    'this return list of requirements'

    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name="mlproject",
    version="0.0.1",
    author="vivek",
    author_email="vivekchouhan2512@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)