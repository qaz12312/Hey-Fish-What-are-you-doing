# log format for debug
import os
import sys
from time import strftime

# 設定目錄
dir_path = os.path.dirname(os.path.abspath(__file__)) + '/debug/'

# 若不存在則新建
if not os.path.isdir(dir_path):
    os.makedirs(dir_path)

# -----------------------------------------------------------
# Write a log
# -----------------------------------------------------------
def writeLog(programInfo, action, message):
    with open(dir_path + action, 'w') as f:
        f.write("Called from line {}".format(programInfo["lineNum"]))
        if programInfo["funName"] != False:
            f.write(", in {}()".format(programInfo["funName"]))
        f.write(", {}.\n\n-------------------------------------------------------\n\n".format(programInfo["fileName"]))
        # write log
        for key,val in message.items():
            f.write("{} : {}\n".format(key,val))
        
        f.write("\n-------------------------------------------------------\n\nFinish write in {}.".format(strftime("%Y-%m-%d %H:%M:%S")))


# -----------------------------------------------------------
# Get FishDebug.py info
# -----------------------------------------------------------
def help():
    msg = """    
    writeLog() args:
        programPath: dict, The absolute path of the file & The line where the function is called and function Name ({"lineNum":int, "funName":False|string, "fileName":string})
        action     : string, file name which you want to write in (There can be no blanks, please use _ instead)
        message    : dict,   message
    """
    print(msg)


# -----------------------------------------------------------
# Test for each function
# -----------------------------------------------------------
def testFunc():
    print("test writeLog().")
    writeLog({"lineNum":47, "funName":"testFunc", "fileName":"C:/Users/88692/Desktop/Hey-Fish-What-are-you-doing/FishDebug.py"},"Test_WriteLog",{'apple':1,'banana':2})
    print("finish.")


if __name__ == '__main__':
    if len(sys.argv) == 2 :
        option = sys.argv[1]
        if option == '-h' or option == '--help': # see help message
            help()
            sys.exit(0)
        elif option == '--test':
            testFunc()
            sys.exit(0)
    
    print_help = """
    Description:
    List commands

    Usage:
    FishDebug.py [options]

    Options:
    -h, --help  Display function args info
        --test  Run test for each function
    """
    print(print_help)