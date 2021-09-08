"""
Log format for debug mode.
"""
from os import getenv, makedirs
from os.path import isdir,split
import sys
from time import strftime
from dotenv import load_dotenv

load_dotenv()
# 設定目錄
dir_path = getenv('PROJECT_PATH') + '/debug/'

# 若不存在則新建
if not isdir(dir_path):
    makedirs(dir_path)

# -----------------------------------------------------------
# Write a log
# -----------------------------------------------------------
def writeLog(programInfo, file, message):
    """
    Write a log.

    Parameters
    ----------
    programInfo : `dict[str, Any]`
        The absolute path of the file & The line where the function is called and function Name (`{"lineNum":int, "funName":False|string, "fileName":string}`).
    file : `str`
        File name which you want to write in (There can be no blanks, please use `_` instead).
    message : `Any`
    """
    head = (split(file))[0]
    if not isdir(dir_path + head):
        makedirs(dir_path + head)
    
    with open(dir_path + file + '.log', 'w') as f:
        f.write("Called from line {}".format(programInfo["lineNum"]))
        if programInfo["funName"] != False:
            f.write(", in {}()".format(programInfo["funName"]))
        f.write(", {}.\n\n-------------------------------------------------------\n\n".format(programInfo["fileName"]))
        # write log
        if type(message).__name__ == 'list' or type(message).__name__ == 'ndarray':
            idx = 0
            for value in message:
                f.write("[{}] = \n".format(idx))

                if type(value).__name__ == 'list' or type(value).__name__ == 'ndarray':
                    i = 0
                    for val in value:
                        f.write("\t[{}] = {}\n".format(i,val))
                        i += 1
                
                elif type(value).__name__ == 'dict':
                    for key,val in value.items():
                        if key == "ONLY VALUE":
                            f.write("\t{}\n".format(val))
                        else:
                            f.write("\t{} : {}\n".format(key,val))
                
                else:
                    f.write("\t{}\n".format(value))
                
                idx += 1

        elif type(message).__name__ == 'dict':
            for key,val in message.items():
                if key == "ONLY VALUE":
                    f.write("{}\n".format(val))
                else:
                    f.write("{} : {}\n".format(key,val))
        
        else:
            f.write("{}\n".format(message))
        
        f.write("\n-------------------------------------------------------\n\nFinish write in {}.".format(strftime("%Y-%m-%d %H:%M:%S")))


# -----------------------------------------------------------
# Get FishDebug.py info
# -----------------------------------------------------------
def help():
    """
    Get FishDebug.py info.
    """
    msg = """    
    writeLog() args:
        programPath: dict, The absolute path of the file & The line where the function is called and function Name ({"lineNum":int, "funName":False|string, "fileName":string})
        file     : string, File name which you want to write in (There can be no blanks, please use _ instead)
        message    : Any, Message
    """
    print(msg)


# -----------------------------------------------------------
# Test for each function
# -----------------------------------------------------------
def testFunc():
    """
    Test for each function.
    """
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