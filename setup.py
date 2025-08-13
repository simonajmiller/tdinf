from setuptools import setup

setup(
    name='tdinf',
    version='0.0dev',
    packages=['tdinf',
              'tdinf.utils',
              'tdinf.pipe'
              ],
    install_requires=[],
    scripts=['tdinf/run_sampler.py',
             'tdinf/waveform_h5s.py', 
             'tdinf/group_postprocess.py',
             'pipe/tdinf_condor_pipe.py',
             'pipe/tdinf_slurm_pipe.py',
             ],
    license='GPL',
    long_description=open('README.md').read(),
)

