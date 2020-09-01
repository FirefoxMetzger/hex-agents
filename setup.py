from setuptools import setup, find_packages

setup(
    name='hex_agents',
    version='0.0.0',
    url='https://github.com/FirefoxMetzger/hex-agents',
    author='Sebastian Wallkotter (FirefoxMetzger)',
    author_email='sebastian.wallkotter@it.uu.se',
    description='Various agents that can learn how to play the hex boardgame',
    packages=find_packages(),
    install_requires=[
        'numpy >= 1.18.2',
        'matplotlib >= 3.2.1',
        'tensorflow >= 2.2.0'
    ]
)
