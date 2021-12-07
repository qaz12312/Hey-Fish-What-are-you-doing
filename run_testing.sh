#!/bin/bash

# cd pwd
cd `dirname $0`

read -p '是否載入 Deeplabcut 的全部檔案至 CSV/test 資料夾下? (type:y/n) ' is_load
while [ "$is_load" != 'y' ] && [ "$is_load" != 'n' ]
do
    read -p $'[!警告]只能輸入 y 或是 n\n是否載入 Deeplabcut 的全部檔案至 CSV/test 資料夾下? (type:y/n) ' is_load
done
if [ "$is_load" = 'y' ]
then
    read -p '是否清空在 CSV/test 資料夾裡的先前資料? (type:y/n) ' is_clear
    while [ "$is_load" != 'y' ] && [ "$is_load" != 'n' ]
    do
        echo "$(tput setaf 5)[!警告]只能輸入 y 或是 n$(tput sgr 0)"
        read -p '是否清空在 CSV/test 資料夾裡的先前資料? (type:y/n) ' is_clear
    done
    if [ "$is_clear" = 'y' ]
    then
        rm -rf ./CSV/test/*
    fi

    test_dir_path='/media/ntou501/4f2b9da8-a755-49a3-afea-60704f1a7d00/merge/mergeFish0816-cse509-2021-08-16/testing'
    read -p "要載入的資料夾路徑是否為: ${test_dir_path}? (type:y/n) " is_right
    while [ "$is_right" != 'y' ] && [ "$is_right" != 'n' ]
    do
        echo "$(tput setaf 5)[!警告]只能輸入 y 或是 n$(tput sgr 0)"
        read -p "要載入的資料夾路徑是否為: ${test_dir_path}? (type:y/n) " is_right
    done
    if [ "$is_right" = 'n' ]
    then
        while [ "$is_right" != 'y' ]
        do
            read -p "請輸入要載入的資料夾路徑: " test_dir_path
            read -p "要載入的資料夾路徑是否為: ${test_dir_path}? (type:y/n) " is_right
        done
    fi

    echo -e '\033[30m\e[1;43m正在載入資料中.....\e[0m'
    if [ ! -d ./CSV/test ];then
        mkdir ./CSV/test
    fi

    file_count=0
    if [ -d "$test_dir_path" ]  # 檢查是否為目錄
    then
        file_names=($(ls -d "$test_dir_path"*.csv))
        for file in "${file_names[@]}"
        do
            cp "$file" ./CSV/test # copy file
            echo "$test_dir_path$file.....成功"
            file_count=$(($file_count + 1))
        done
        echo "[./CSV/test] 總共成功載入 $file_count 個檔案"
    else
        echo "$(tput setaf 5)[!警告] $test_dir_path 目錄不存在$(tput sgr 0) 請重新進行操作"
        exit 0
    fi
fi

echo -e "\n**********************************************************"
echo -e "\033[30m\e[1;43mConvert data.....\e[0m"

if [ -f "./convertTo_txt/X_test.txt" ]
then
    truncate -s 0 "convertTo_txt/X_test.txt" # empty content of the file
    echo "clear convertTo_txt/X_test.txt.....success"
fi

echo -e "\e[1;34m建立數據\e[0m"
/usr/bin/python3.6 convertData.py --type 1

echo "**********************************************************"
echo -e "\033[30m\e[1;43mTesting data.....\e[0m"
/usr/bin/python3.6 testing.py

echo "**********************************************************"
echo "$(tput setaf 0)$(tput setab 2) 結束 $(tput sgr 0)"
exit 0
