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
    declare -A label_dir=( ['f']='forage_test/' ['n']='normal_test/' ['s']='sleep_test/')
    label_dir_path='/media/ntou501/4f2b9da8-a755-49a3-afea-60704f1a7d00/merge/mergeFish0816-cse509-2021-08-16/'
    read -p $'載入幾種類別的檔案? (type:2/3) ' n_label
    while [ "$n_label" != '2' ] && [ "$n_label" != '3' ]
    do
       read -p $'[!警告]只能輸入 2 或是 3\n載入幾種類別的檔案? (type:2/3) ' n_label
    done
    if [ "$n_label" = '2' ]
    then
        read -p '要載入的第一種檔案類別為? (type:n/s/f) ' label[0]
        while [ "${label[0]}" != 'n' ] && [ "${label[0]}" != 's' ] && [ "${label[0]}" != 'f' ];do read -p $'[!警告]只能輸入 n/s/f 這三者之一\n要載入的第一種檔案類別為? (type:n/s/f) ' label[0]; done;
        read -p '要載入的第二種檔案類別為? (type:n/s/f) ' label[1]
        while [ "${label[0]}" == "${label[1]}" ] || ([ "${label[1]}" != 'n' ] && [ "${label[1]}" != 's' ] && [ "${label[1]}" != 'f' ])
        do
            if [ "${label[0]}" == "${label[1]}" ]
            then
                echo '[!警告]不可為同一類別'
            else
                echo '[!警告]只能輸入 n/s/f 這三者之一'
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

    echo -e '\e[1;43mLoading all croods files.....\e[0m'
    if [ ! -d ./CSV/test ];then
        mkdir ./CSV/test
    fi
    if [ ! -d ./CSV/train ];then
        mkdir ./CSV/train
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
                if (( $file_count >= 6 )) # modify
                then
                    target_dir='test'
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
lstm_files=("X_test.txt" "X_train.txt" "Y_test.txt" "Y_train.txt")
read -p '是否刪除 convertTo_txt 資料夾下的所有檔案? (type:y/n) ' is_remove
while [ "$is_remove" != 'y' ] && [ "$is_remove" != 'n' ]
do
    read -p $'[!警告]只能輸入 y 或是 n\n是否刪除 convertTo_txt 資料夾下的所有檔案? (type:y/n) ' is_remove
done
if [ "$is_remove" = 'y' ]
then
    echo -e "\e[1;34mRemove all lstm files: yes\e[0m"
    for file in "${lstm_files[@]}"
    do
        truncate -s 0 "convertTo_txt/$file" # empty content of the file
        echo "clear convertTo_txt/$file.....success"
    done
else
    echo -e "\e[1;34mRemove all lstm files: no\t資料會接續寫下去\e[0m"
fi
echo -e "\e[1;43mConvert data.....\e[0m"
/usr/bin/python3.6 convertData.py
echo "**********************************************************"
read -p '執行幾種 hidden cell (從 1層 開始) ? ' hidden_types
while ! [[ "$hidden_types" =~ ^[1-9][0-9]*$ ]]
do
    read -p $'[!警告]只能輸入大於 1 的正整數\n執行幾種 hidden cell (從 1層 開始) ? ' hidden_types
done
echo -e "\e[1;43mTraining data.....\e[0m"
for n_hidden in $(seq 1 $hidden_types);
do
    echo -e "\e[1;34mRun hidden cell= $n_hidden\e[0m"
    /usr/bin/python3.6 training.py --hidden $n_hidden
done

echo "**********************************************************"
echo -e "\e[1;42m 結束 \e[0m"
exit 0
