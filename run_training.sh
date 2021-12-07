#!/bin/bash

# cd pwd
cd `dirname $0`

read -p '是否載入 Deeplabcut 的全部檔案至 CSV 資料夾下? (type:y/n) ' is_load
while [ "$is_load" != 'y' ] && [ "$is_load" != 'n' ]
do
    read -p $'[!警告]只能輸入 y 或是 n\n是否載入 Deeplabcut 的全部檔案至 CSV 資料夾下? (type:y/n) ' is_load
done
if [ "$is_load" = 'y' ]
then
    read -p '是否清空在 CSV 資料夾裡的先前資料? (type:y/n) ' is_clear
    while [ "$is_clear" != 'y' ] && [ "$is_clear" != 'n' ]
    do
        echo "$(tput setaf 5)[!警告]只能輸入 y 或是 n$(tput sgr 0)"
        read -p '是否清空在 CSV 資料夾裡的先前資料? (type:y/n) ' is_clear
    done
    if [ "$is_clear" = 'y' ]
    then
        rm -rf ./CSV/*
    fi

    declare -A label_dir=( ['f']='forage_test/' ['n']='normal_test/' ['s']='sleep_test/')
    label_dir_path='/media/ntou501/4f2b9da8-a755-49a3-afea-60704f1a7d00/merge/mergeFish0816-cse509-2021-08-16/'
    label=('n' 's' 'f')

    echo -e "\e[1;34m要載入的類別檔案資料夾: \e[0m"
    for key in "${label[@]}"
    do
        echo -e "\e[1;34m$key ${label_dir[$key]}\e[0m"
    done

    echo -e '\033[30m\e[1;43mLoading all croods files.....\e[0m'
    if [ ! -d ./CSV/train ];then
        mkdir ./CSV/train
    fi
    if [ ! -d ./CSV/validate ];then
        mkdir ./CSV/validate
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
                cp "$file" "CSV/$target_dir" # copy file
                echo "loading $file.....success"
                file_count=$(($file_count + 1))
            done
            echo "[./$directory] 總共有 $file_count 個檔案 成功"
        else
            echo "$abs_dir_path 目錄不存在"
        fi
        total_files=$(($total_files + $file_count))
    done
    echo -e "\n\e[1;32m總共有 $total_files 個檔案 成功\e[0m"
fi

echo -e "\n**********************************************************"
lstm_files=("X_validate.txt" "X_train.txt" "Y_validate.txt" "Y_train.txt")
read -p '是否刪除 convertTo_txt 資料夾下的檔案? (type:y/n) ' is_remove
while [ "$is_remove" != 'y' ] && [ "$is_remove" != 'n' ]
do
    read -p $'[!警告]只能輸入 y 或是 n\n是否刪除 convertTo_txt 資料夾下的檔案? (type:y/n) ' is_remove
done
if [ "$is_remove" = 'y' ]
then
    echo -e "\e[1;34mRemove all lstm files: yes 資料會重新填寫\e[0m"
    for file in "${lstm_files[@]}"
    do
        if [ -f "./convertTo_txt/$file" ]
        then
            truncate -s 0 "convertTo_txt/$file" # empty content of the file
            echo "clear convertTo_txt/$file.....success"
        fi
    done
else
    echo -e "\e[1;34mRemove all lstm files: no 資料會接續寫下去\e[0m"
fi
echo -e "\033[30m\e[1;43mConvert data.....\e[0m"
/usr/bin/python3.6 convertData.py --type 2
echo "**********************************************************"
echo -e "\033[30m\e[1;43mTraining data.....\e[0m"
/usr/bin/python3.6 training.py

echo "**********************************************************"
echo "$(tput setaf 0)$(tput setab 2) 結束訓練 $(tput sgr 0)"
exit 0
