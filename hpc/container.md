# Running Containers with Apptainer & Docker

In this part, we briefly show you how to run your containers (like [Nvidia Containers](https://catalog.ngc.nvidia.com/containers)) on the HPC with [Apptainer]((https://apptainer.org/docs/user/latest/)).

No modules need to be loaded in order to run containers.

First of all, it is important to consider that Singularity is not in use anymore and Apptainer is the one replacing it. 
If for some reason you need Singularity you may install it locally in your directory or contact HPC for ordering the installation of this tool.

As follows, we show how  to run a container step by step.

Apptainer is able to run existing docker images once they are converted into a .sif file, but it cannot build a container directly from a dockerfile.
To get around this, you can either directly get an existing docker image from a docker repo (eg dockerhub or NVIDIA's container catalog) and convert it to a sif file:

```bash
apptainer pull tensorflow-19.11-tf1-py3.sif docker://nvcr.io/nvidia/tensorflow:19.11-tf1-py3
```

Or you can build a docker container locally on your computer, then move it to the cluster as shown here:

```bash
#whilst on your local computer in the same directory as an existing dockerfile
docker build . -t name_of_the_image # --tag , -t Name and optionally a tag in the 'name:tag' format
```

to see the images you have on docker:

```
docker images
```

From that you can see the image_id that you are going to use for the following command to save the image.


```bash
sudo docker save image_id -o file_name.tar
```

Then, you move the file to the server. You can do this using scp/rsync or similar

Next, you need to build the singularity image from the file like follows:

```bash
apptainer build container_name.sif docker-archive://file_name.tar
```

**Note**: We use docker-archive:// before the name of our file to tell the Apptainer that it is a docker image.

After either building your .sif file from a docker-archive or by pulling it from a docker repo, you can then interact with the container.

For example, the following command will run the image:

```
 apptainer run --nv -B /home/common/datasets/imagenet2012/:/imagenet ResNet50.sif 
 
 # --nv is for enabling Nvidia support, therefore GPUs
 # -B or --bind is for mounting - here we are mounting the imagenet dataset on the centralstorage into the container under the /imagenet mountpoint
```

If it was successful, you can launch your tasks from inside the container.

Finally, our job script will look like as follows (it shows how you can run a Nvidia container from a SLURM job script):

```bash
#!/bin/bash

#SBATCH --job-name=resnet-container-single-gpu          # Job name
#SBATCH --output=resnet-container-single-gpu.out
#SBATCH --cpus-per-task=8                               # Schedule 8 cores (includes hyperthreading)
#SBATCH --gres=gpu                                      # Schedule a GPU, it can be on 2 gpus like gpu:2
#SBATCH --gres-flags=enforce-binding                    # Get directoy connected cores to GPU
#SBATCH --time=23:59:59                                 # Run time (hh:mm:ss) - run for one hour max
#SBATCH --partition=red                                 # Run on either the Red or Brown queue
##SBATCH --exclusive                                     # Exclusivly getting Resources
#SBATCH --nodelist=cn5                                  # The requested Server


echo "Running on $(hostname):"                          # showing which node it is running on

apptainer run --nv -B /home/common/datasets/imagenet2012/:/imagenet ResNet50.sif python models/DeepLearningExamples/PyTorch/Classification/ConvNets/main.py --arch resnet50 --epochs 5 /imagenet/
```