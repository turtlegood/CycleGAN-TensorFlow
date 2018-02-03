full_image_size=160
common_arg="\
        --face_model_path /home/rail/TomChen/Others/CelebaData/facenet_ms.pb \
        --full_image_size $full_image_size \
        --eye_image_size 48 \
        --eye_y 70 \
        --lambda_face 0 \
        --use_G_skip_conn False"

# $1 => default value; $2 => value
function chkpt_dir_prefix {
    if [ -z "$2" ];
    then
        if [ "$1" == "formal" ]
        then
            echo "checkpoints"
        else
            echo "checkpoints_informal"
        fi
    else
        if [ "$2" == "formal" ]
        then
            echo "checkpoints"
        else
            echo "checkpoints_informal"
        fi
    fi
}

function ls_chkpt {
    ls ./$(chkpt_dir_prefix formal $1)/ -1 -r
}

# $1 => idx; $2 => formal
function chkpt_from_idx {
    result=$(ls_chkpt $2 | sed $1'q;d')
    echo $result
}

case "$1" in 
ls )
    echo "mode: $(chkpt_dir_prefix formal $2)"
    ls_chkpt $2 | nl
    ;;
prebuild_data | prebuild )
    python3 prebuild_data.py \
        --base_path /home/rail/TomChen/Others/CelebaData \
        --postfix not has
    ;;
build_data )
    # mv ./data/crop_has/output/ ./data/crop_has_aug/
    python3 build_data.py \
        --X_input_dir /home/rail/TomChen/Others/CelebaData/crop_has \
        --Y_input_dir /home/rail/TomChen/Others/CelebaData/crop_not \
        --X_output_file /home/rail/TomChen/Others/CelebaData/has.tfrecords \
        --Y_output_file /home/rail/TomChen/Others/CelebaData/not.tfrecords
    ;;
train )
    if test "$#" -ne 2;
    then
        addition=""
    else 
        if [ "$3" == "formal" ]
        then
            echo 'formal'
            addition="--formal True"
        else
            chkpt=$(chkpt_from_idx $2 $3)
            echo 'chkpt' $chkpt
            addition="--load_model $chkpt"
        fi
    fi
    cmd="python3 train.py \
        --X /home/rail/TomChen/Others/CelebaData/has.tfrecords \
        --Y /home/rail/TomChen/Others/CelebaData/not.tfrecords \
        --batch_size 1 \
        $common_arg \
        $addition"
    echo "cmd" $cmd; $cmd
    ;;
export )
    if test "$#" -ne 2;
    then
        echo "need param 2 to be folder name"
    else 
        chkpt=$(chkpt_from_idx $2 $3)
        echo 'chkpt' $chkpt
        python3 export_graph.py \
            --name "$chkpt" \
            $common_arg
    fi
    ;;
inference | infer )
    chkpt=$(chkpt_from_idx $2 $3)
    echo 'chkpt' $chkpt
    python3 inference.py \
        --model_dir pretrained/$chkpt \
        --input_dir test/input \
        --output_dir test/$chkpt \
        --full_image_size $full_image_size
    ;;
log )
    chkpt=$(chkpt_from_idx $2 $3)
    cmd="tensorboard --logdir /home/rail/TomChen/Sync/CycleGAN-TensorFlow/$(chkpt_dir_prefix informal $3)/$chkpt"
    echo "cmd" $cmd; $cmd
    ;;
log_multi )
    logdir=$(echo $2 | sed 's,:,:/home/rail/TomChen/Sync/CycleGAN-TensorFlow/checkpoints/,g')
    tensorboard --logdir $logdir
    ;;
* )
    echo "Wrong input"
esac