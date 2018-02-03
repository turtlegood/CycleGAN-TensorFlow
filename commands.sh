case "$1" in 
prebuild_data | prebuild )
    python3 prebuild_data.py \
        --base_path ~/TomChen/Others/CelebaData \
        --postfix not has
    ;;
build_data )
    # mv ./data/crop_has/output/ ./data/crop_has_aug/
    python3 build_data.py \
        --X_input_dir ~/TomChen/Others/CelebaData/crop_has \
        --Y_input_dir ~/TomChen/Others/CelebaData/crop_not \
        --X_output_file ~/TomChen/Others/CelebaData/has.tfrecords \
        --Y_output_file ~/TomChen/Others/CelebaData/not.tfrecords
    ;;
train_old )
    python3 train.py \
        --X ~/TomChen/Others/CelebaData/old_has.tfrecords \
        --Y ~/TomChen/Others/CelebaData/old_not.tfrecords \
        --image_size 48
    ;;
train_new )
    if test "$#" -ne 2;
    then
        addition=""
    else 
        addition="--load_model $2"
    fi
    python3 train.py \
        --X ~/TomChen/Others/CelebaData/has.tfrecords \
        --Y ~/TomChen/Others/CelebaData/not.tfrecords \
        --face_model_path ~/TomChen/Others/CelebaData/facenet_ms.pb \
        --full_image_size 160 \
        --eye_image_size 48 \
        --eye_y 70 \
        --use_G_skip_conn True \
        --lambda_face 0.01 \
        --batch_size 1 \
        $addition
    ;;
export )
    for D in `find ./checkpoints/ -mindepth 1 -maxdepth 1 -type d`
    do
        echo exporting: $D
        short="${D#./checkpoints/}"
        python3 export_graph.py \
            --checkpoint_dir checkpoints/$short \
            --XtoY_model add-$short.pb \
            --YtoX_model rem-$short.pb \
            --image_size 48
    done
    ;;
* )
    echo "Wrong input"
esac