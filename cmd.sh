full_image_size=160
common_arg="\
        --face_model_path ~/TomChen/Others/CelebaData/facenet_ms.pb \
        --full_image_size $full_image_size \
        --eye_image_size 48 \
        --eye_y 70 \
        --lambda_face 0.01 \
        --use_G_skip_conn True"
ls_chkpt="ls ./checkpoints/ -1 -r"

function chkpt_from_idx {
    result=$($ls_chkpt | sed $1'q;d')
    echo $result
}

case "$1" in 
ls )
    $ls_chkpt | nl
    ;;
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
train_new )
    if test "$#" -ne 2;
    then
        addition=""
    else 
        chkpt=$(chkpt_from_idx $2)
        echo 'chkpt' $chkpt
        addition="--load_model $chkpt"
    fi
    python3 train.py \
        --X ~/TomChen/Others/CelebaData/has.tfrecords \
        --Y ~/TomChen/Others/CelebaData/not.tfrecords \
        --batch_size 1 \
        $common_arg \
        $addition
    ;;
export )
    if test "$#" -ne 2;
    then
        echo "need param 2 to be folder name"
    else 
        chkpt=$(chkpt_from_idx $2)
        echo 'chkpt' $chkpt
        python3 export_graph.py \
            --name "$chkpt" \
            $common_arg
    fi
    ;;
inference | infer )
    if test "$#" -ne 2;
    then
        echo "need param 2 to be folder name"
    else 
        chkpt=$(chkpt_from_idx $2)
        echo 'chkpt' $chkpt
        python3 inference.py \
            --model_dir pretrained/$chkpt \
            --input_dir test/input \
            --output_dir test/$chkpt \
            --full_image_size $full_image_size
    fi
    ;;
log_single )
    tensorboard --logdir /home/rail/TomChen/Sync/CycleGAN-TensorFlow/checkpoints/$2
    ;;
log_multi )
    logdir=$(echo $2 | sed 's,:,:/home/rail/TomChen/Sync/CycleGAN-TensorFlow/checkpoints/,g')
    tensorboard --logdir $logdir
    ;;
* )
    echo "Wrong input"
esac