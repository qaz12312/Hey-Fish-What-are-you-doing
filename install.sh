#!/bin/bash
# 用來定義你要使用的 shell

# cd pwd
cd `dirname $0`

# Check root
echo "Check root......"
logname=$(env | grep LOGNAME | sed 's|LOGNAME=||')
if [ $logname != "ntou501" ] && [ "$name" != "root" ]; then
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
version_arr=(${version//./ }) 
if [ "${version_arr[0]}" != "18" ]; then
    echo 'This OS version is not 18.'
    exit 1
fi
echo $version
echo "**********************************************************"

# Install python3.6
echo "Install python3.6......"
# sudo apt install python3 python3-libs -y
# sudo apt install python3-setuptools -y
# sudo apt install python3-pip -y
echo "**********************************************************"

# Install python3 Module
echo "Install Python3 Module......"
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

echo "\nAll steps have been done.\n"
exit 0