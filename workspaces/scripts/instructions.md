### Sequence:
1 - Gather the sample images with their associated labels
2 - Modify the paths in the conversion.py and DrawBoxes.ipynb
3 - Run conversion.py to generate a formatted csv file
4 - Run DrawBoxes.ipynb (image scaling section) against the formatted csv file
    this will generate two files, test and train
5 - Run generate_tfrecord.py and pass in the test file and output file, repeat for train file
    this will generate the protobufs which can be used directly in the tensorflow object detection training
6 - If using a pre-trained model, download the appropriate one from model zoo and place in the checkpoint directory. This model was initally trained using the same model as the CarND Object Detection Laboratory (ssd_mobilnet_v1 ... ).
