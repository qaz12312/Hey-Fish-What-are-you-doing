#!/bin/bash
# 用來定義你要使用的 shell

# cd pwd
cd `dirname $0`

# Check root
echo "Check root......"
logname=$(env | grep LOGNAME | sed 's|LOGNAME=||')
if [ "$logname" != "ntou501" ] && [ "$logname" != "root" ]; then
    echo 'This user is not root, please change user with `su -` command.(Notice:Use `su` command can`t be installed correctly.)'
    exit 1
fi

# Check OS
echo "Check OS......"
name=$(more /etc/os-release | grep "^NAME=" | sed 's|NAME=||' | sed 's/\"//g')
version=$(more /etc/os-release | grep "^VERSION_ID=" | sed 's|VERSION_ID=||' | sed 's/\"//g')
if [ "$name" != "Ubuntu" ]; then
    echo 'This OS is not [Ubuntu].'
    exit 1
fi
echo $name
version_arr=(${version//./ }) # ${parameter//pattern/string}: 用string來替換parameter變數中所有匹配的pattern
if [ "${version_arr[0]}" != "18" ]; then
    echo 'This OS version is not 18.'
    exit 1
fi
echo $version
echo "**********************************************************"

# Install python3.6
echo -e "\e[1;43mInstall python3.6......\e[0m"
# sudo apt install python3 python3-libs -y
# sudo apt install python3-setuptools -y
# sudo apt install python3-pip -y
echo "**********************************************************"

# Install python3 Module
echo -e "\e[1;43mInstall Python3 Module......\e[0m"
/usr/bin/python3.6 -m pip install -r requirements.txt

# 確認版本是1.15.0
echo "Check tensorflow version......"
tf_version=$(/usr/bin/python3.6 -m pip show tensorflow | awk '/^Version: / {sub("^Version: ", ""); print}')
if [ "$tf_version" != "1.15.0" ]; then
    echo "This OS version is not 1.15.0."
    exit 1
fi
echo $tf_version
echo "**********************************************************"

echo -e "\e[1;42m All steps have been done. \e[0m"
exit 0