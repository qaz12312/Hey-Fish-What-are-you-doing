#!/bin/bash
function run_codes(){
    lstm_files=("X_test.txt" "X_train.txt" "X_validate.txt" "Y_train.txt" "Y_validate.txt")
    echo -e "\e[1;34mRemove all LSTM data files: yes 資料會重新填寫\e[0m"
    for file in "${lstm_files[@]}"
    do
        if [ -f "./TESTING/convertTo_txt/$file" ]
        then
            truncate -s 0 "TESTING/convertTo_txt/$file" # empty content of the file
            echo "clear TESTING/convertTo_txt/$file.....success"
        fi
    done
    
    echo -e "\033[30m\e[1;43mConvert data.....\e[0m"
    hidden_types=$1
    n_step=$2
    n_jump=$3
    is_normal=$4
    if [ "$is_normal" == 1 ]
    then
        /usr/bin/python3.6 TESTING_convertData.py --steps $n_step --jumps $n_jump --normal $is_normal --len $5 --translation $6 --rotate $7
    else
        /usr/bin/python3.6 TESTING_convertData.py --steps $n_step --jumps $n_jump --normal $is_normal
    fi
    result=$?
    
    if [ "$result" == 0 ] 
    then
        echo "**********************************************************"
        echo -e "\033[30m\e[1;43mTraining data.....\e[0m"
        for n_hidden in $(seq 1 $hidden_types);
        do
            echo -e "\e[1;34mRun hidden cell= $n_hidden\e[0m"
            /usr/bin/python3.6 TESTING_training.py --hidden $n_hidden --steps $n_step
        done
    fi
}

# cd pwd
cd `dirname $0`

read -p '是否載入 Deeplabcut 的全部檔案至 TESTING/CSV 資料夾下? (type:y/n) ' is_load
while [ "$is_load" != 'y' ] && [ "$is_load" != 'n' ]
do
    echo "$(tput setaf 5)[!警告]只能輸入 y 或是 n$(tput sgr 0)"
    read -p '是否載入 Deeplabcut 的全部檔案至 TESTING/CSV 資料夾下? (type:y/n) ' is_load
done

if [ "$is_load" = 'y' ]
then
    read -p '是否清空在 TESTING/CSV 資料夾裡的先前資料? (type:y/n) ' is_clear
    while [ "$is_clear" != 'y' ] && [ "$is_clear" != 'n' ]
    do
        echo "$(tput setaf 5)[!警告]只能輸入 y 或是 n$(tput sgr 0)"
        read -p '是否清空在 TESTING/CSV 資料夾裡的先前資料? (type:y/n) ' is_clear
    done
    if [ "$is_clear" = 'y' ]
    then
        rm -rf ./TESTING/CSV/*
    fi
    
    declare -A label_dir=( ['f']='forage_test/' ['n']='normal_test/' ['s']='sleep_test/')
    label_dir_path='/media/ntou501/4f2b9da8-a755-49a3-afea-60704f1a7d00/merge/mergeFish0816-cse509-2021-08-16/'
    
    read -p '載入幾種類別的檔案? (type:2/3) ' n_label
    while [ "$n_label" != '2' ] && [ "$n_label" != '3' ]
    do
        echo "$(tput setaf 5)[!警告]只能輸入 2 或是 3$(tput sgr 0)"
        read -p '載入幾種類別的檔案? (type:2/3) ' n_label
    done
    echo -e "\e[1;31m請確保 TESTING_training.py 的 LABEL_LIST的元素總數 等同於您所輸入的值:\033[47m $n_label \e[0m"
    sleep 2
    if [ "$n_label" = '2' ]
    then
        read -p '要載入的第一種檔案類別為? (type:n/s/f) ' label[0]
        while [ "${label[0]}" != 'n' ] && [ "${label[0]}" != 's' ] && [ "${label[0]}" != 'f' ]
        do
            echo "$(tput setaf 5)[!警告]只能輸入 n/s/f 這三者之一$(tput sgr 0)"
            read -p '要載入的第一種檔案類別為? (type:n/s/f) ' label[0]
        done
        read -p '要載入的第二種檔案類別為? (type:n/s/f) ' label[1]
        while [ "${label[0]}" == "${label[1]}" ] || ([ "${label[1]}" != 'n' ] && [ "${label[1]}" != 's' ] && [ "${label[1]}" != 'f' ])
        do
            if [ "${label[0]}" == "${label[1]}" ]
            then
                echo "$(tput setaf 5)[!警告]不可為同一類別$(tput sgr 0)"
            else
                echo "$(tput setaf 5)[!警告]只能輸入 n/s/f 這三者之一$(tput sgr 0)"
            fi
            read -p '要載入的第二種檔案類別為? (type:n/s/f) ' label[1]
        done
    else
        label=('n' 's' 'f')
    fi
    echo -e "\e[1;34m要載入的類別檔案資料夾: \e[0m"
    for key in "${label[@]}"
    do
        echo -e "\e[1;34m$key ${label_dir[$key]}\e[0m"
    done

    echo -e '\033[30m\e[1;43mLoading all croods files.....\e[0m'
    if [ ! -d ./TESTING/CSV/train ];then
        mkdir ./TESTING/CSV/train
    fi
    if [ ! -d ./TESTING/CSV/validate ];then
        mkdir ./TESTING/CSV/validate
    fi
    if [ ! -d ./TESTING/CSV/test ];then
        mkdir ./TESTING/CSV/test
    fi
    total_files=0
    for key in "${label[@]}"
    do
        directory=${label_dir[$key]}
        abs_dir_path=$label_dir_path$directory
        file_count=0
        if [ -d "$abs_dir_path" ]  # 檢查是否為目錄
        then
            file_names=($(ls -d "$abs_dir_path"*.csv))
            for file in "${file_names[@]}"
            do
                if [ $(expr $file_count % 3) -eq 2 ] # three
                then
                    target_dir='validate'
                else
                    target_dir='train'
                fi
                cp "$file" "TESTING/CSV/$target_dir" # copy file
                echo "loading $file.....success"
                file_count=$(($file_count + 1))
            done
            echo "[./$directory] 總共有 $file_count 個檔案 成功"
        else
            echo "$(tput setaf 7)$(tput setab 1)[!異常]Deeplabcut目錄 $abs_dir_path 不存在$(tput sgr 0)"
            exit 1
        fi
        total_files=$(($total_files + $file_count))
    done
    echo -e "\n\e[1;32m總共有 $total_files 個檔案 成功\e[0m"
fi

echo -e "\n**********************************************************"

N_STEPS=(15 20 25 30 35)
N_JUMP=(2 5 10 15 25)
N_LEN=(0 1 2 3 4 5)

read -p '執行幾種 hidden cell (從 1層 開始) ? ' hidden_types
while ! [[ "$hidden_types" =~ ^[1-9][0-9]*$ ]]
do
    echo "$(tput setaf 5)[!警告]只能輸入大於 1 的正整數$(tput sgr 0)"
    read -p '執行幾種 hidden cell (從 1層 開始) ? ' hidden_types
done

echo "**********************************************************"

for n_step in "${N_STEPS[@]}"
do
    for n_jump in "${N_JUMP[@]}"
    do
        run_codes $hidden_types $n_step $n_jump 0
        for n_len in "${N_LEN[@]}"
        do
            if [ "$n_len" != 0 ]
            then
                run_codes $hidden_types $n_step $n_jump 1 $n_len 0 0
            fi
            run_codes $hidden_types $n_step $n_jump 1 $n_len 0 1
            run_codes $hidden_types $n_step $n_jump 1 $n_len 0 2
            run_codes $hidden_types $n_step $n_jump 1 $n_len 1 0
            run_codes $hidden_types $n_step $n_jump 1 $n_len 1 1
            run_codes $hidden_types $n_step $n_jump 1 $n_len 1 2
        done
    done
done

echo "**********************************************************"
echo "$(tput setaf 0)$(tput setab 2) 結束 $(tput sgr 0)"
exit 0