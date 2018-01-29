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
    if test "$#" -ne 2;
    then
        addition=""
    else 
        addition="--load_model $2"
    fi
    python3 train.py \
        --X ./data/has.tfrecords \
        --Y ./data/not.tfrecords \
        --face_model_path ./facenet/data/pretrained/ms.pb \
        --full_image_size 160 \
        --eye_image_size 48 \
        --eye_y 70 \
        --lambda_face 0.01 \
        --lambda_pix 1e-6 \
        --batch_size 8 \
        $addition
    ;;
export )
    if test "$#" -ne 2;
    then
        echo "need param 2 to be folder name"
    else 
        python3 export_graph.py \
            --name "$2" \
            --face_model_path ./facenet/data/pretrained/ms.pb \
            --full_image_size 160 \
            --eye_image_size 48 \
            --eye_y 70 \
            --lambda_pix 1e-6
    fi
    ;;
inference | infer )
    if test "$#" -ne 2;
    then
        echo "need param 2 to be folder name"
    else 
    python3 inference.py \
        --model_dir pretrained/$2 \
        --input_dir test/input \
        --output_dir test/$2 \
        --full_image_size 160
    fi
    ;;
* )
    echo "Wrong input"
esac