# psi-2022-accelerator
Pecan Street Inc code for the 2022 PJMF Accelerator

https://www.mcgovern.org/our-work/data-and-society/accelerator/

https://www.pecanstreet.org/

Scott Hinson shinson@pecanstreet.org, 
Stephen Mock smock@pecanstreet.org, 
Jill Harlow jharlow@pecanstreet.org

This is the code and data created for the 2022 PJMF Accelerator by Pecan Street Inc staff.

The src directory contains two subdirectories:

- ann : Contains the `train.py` script that uses the outputs of the notebooks to train and run the model.
- notebooks : the Jupyter notebooks that generate the input space, create the DayCent input files and stage the data on AWS, run the AWS batch jobs, and then collect and collate the output of the jobs into tensor files. These tensor files are the inputs to the `ann/train.py` script. The notebooks themselves contain quite a bit of documentation. The batch job notebook depends on https://github.com/Pecan-Street/daycent-docker-aws which has the containerized version of DayCent to run in the AWS cloud.