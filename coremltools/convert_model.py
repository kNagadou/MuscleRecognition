import coremltools

if __name__ == '__main__':
    coremlmodel = coremltools.converters.caffe.convert(
        ("caffemodel.binaryproto", "deploy.prototxt", "mean.binaryproto"),
        class_labels='labels.json',
        predicted_feature_name='classLabel',
        image_input_names='data',
        image_scale=1.0/255.0
    )

    coremlmodel.save("Muscle.mlmodel")
