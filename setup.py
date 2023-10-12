from setuptools import setup

setup(
    name='time_domain_gw_inference',
    version='0.0dev',
    packages=['time_domain_gw_inference',
              'time_domain_gw_inference.utils'
              ],
    install_requires=[],
    scripts=['time_domain_gw_inference/run_sampler.py'],
    license='GPL',
    long_description=open('README.md').read(),
)

