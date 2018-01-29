case "$1" in 
data | build_data )
    # mv ./data/crop_has/output/ ./data/crop_has_aug/
    python3 build_data.py \
        --X_input_dir ./data/crop_has_aug \
        --Y_input_dir ./data/crop_not_aug \
        --X_output_file ./data/has.tfrecords \
        --Y_output_file ./data/not.tfrecords
    ;;
train )
    python3 train.py \
        --X ./data/has.tfrecords \
        --Y ./data/not.tfrecords \
        --face_model_path ./facenet/data/pretrained/ms.pb \
        --full_image_size 160 \
        --g_image_size 48 \
        --eye_y 70 \
        --lambda_face 0.01
    ;;
export )
    for D in `find ./checkpoints/ -mindepth 1 -maxdepth 1 -type d`
    do
        echo exporting: $D
        short="${D#./checkpoints/}"
        python3 export_graph.py \
            --checkpoint_dir checkpoints/$short \
            --XtoY_model $short.pb \
            --face_model_path ./facenet/data/pretrained/ms.pb \
            --full_image_size 160 \
            --g_image_size 48
    done
    ;;
export_one )
    short="20180129-1050"
    python3 export_graph.py \
        --checkpoint_dir checkpoints/$short \
        --XtoY_model $short.pb \
        --face_model_path ./facenet/data/pretrained/ms.pb \
        --full_image_size 160 \
        --g_image_size 48 \
        --eye_y 70
    ;;
inference | infer )
    python3 inference.py \
        --model checkpoints/20180129-1120/auto-0.pb \
        --input data/test_in.JPEG \
        --output data/test_out.JPEG \
        --full_image_size 160
    ;;
* )
    echo "Wrong input"
esac