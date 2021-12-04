#!/bin/bash

if [ ${#@} -ne 0 ] && [ "${@#"--help"}" = "" ]
then
    echo '...help...'
    echo '選擇環境'
    exit 0
fi

# cd pwd
cd `dirname $0`

declare -A model_dir=( ['0']='正式' ['1']='測試環境')
read -p "$(tput setaf 6)使用哪一種模式?正式0 / 測試1 (type:0/1) $(tput sgr 0)" model
while [ "$model" = '0' ] || ([ "$model" != '0' ] && [ "$model" != '1' ])
do
    if [ "$model" = '0' ]
    then
        read -p "$(tput setaf 1)$(tput setab 7)是否確定進入${model_dir[$model]}模式 [y/n] $(tput sgr 0)" is_sure
        if [ "$is_sure" = 'y' ]
        then
            break
        elif [ "$is_sure" != 'n' ]
        then
            echo -e "$(tput setaf 5)[!警告]只能輸入 y 或是 n\n請重新選擇模式$(tput sgr 0)"
        fi
    else
        echo "$(tput setaf 5)[!警告]只能輸入 0 或是 1$(tput sgr 0)"
    fi
    read -p $"$(tput setaf 6)使用哪一種模式?正式0 / 測試1 (type:0/1) $(tput sgr 0)" model
done
if [ "$model" = '0' ]
then
    echo "$(tput setaf 7)$(tput setab 1)進入${model_dir[$model]}模式.....$(tput sgr 0)"
    # bash run.sh
else
    echo "$(tput setaf 0)$(tput setab 3)進入${model_dir[$model]}模式.....$(tput sgr 0)"
    # bash TESTING_run.sh
fi
exit 0