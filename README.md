# Transportation mode identification for Lifelogging

This repository is the code to train and test deep learning models for Transport Mode Recognition based on the GPS trajectory dataset collected in (Microsoft Research Asia) Geolife project, the GeoLife GPS Trajectories dataset.


### Setting up an environment

#### Using a virtual environment

Usually a virtual environment is used for each program or application of this project. In order to create a virtual environment the following commands can be used inside of app folders:

```bash
virtualenv .
source bin/activate
```

### Requirements

The requirements can be installed using the following command:

> pip install -r requirements.txt



### Dataset 

- GeoLife GPS Trajectories Version 1.3: [geolifdataset](https://msropendata.com/datasets/3b8d1783-2c4f-4bdf-aa58-db777d0fd037)


### Training and Testing Model


```bash
python train.py
```



### Pre-Trained Models

- [Best DeepInsigh + DeepViT](https://drive.google.com/file/d/1WIQa7SmQAIizfRUCDdOh-EP3-z3b74Io/view?usp=drive_link)


- [Best Coords file](https://drive.google.com/file/d/10YL2iLS82VVesxhS1wNvSUhY9G3Fgclo/view?usp=drive_link)

### Demo for transport mode identification using pre-trained model


```bash
python demo.py
```


