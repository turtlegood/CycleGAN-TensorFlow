case "$1" in 
data | build_data )
    # mv ./data/crop_has/output/ ./data/crop_has_aug/
    python3 build_data.py --X_input_dir ./data/crop_has_aug --Y_input_dir ./data/crop_not_aug --X_output_file ./data/has.tfrecords --Y_output_file ./data/not.tfrecords
    ;;
train )
    python3 train.py --X ./data/has.tfrecords --Y ./data/not.tfrecords --image_size 48
    ;;
* )
    echo "Wrong input"
esac