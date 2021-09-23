"""
Log format for debug mode.
"""
from os import getenv, makedirs
from os.path import isdir, split
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
        if type(message).__name__ in ['list', 'ndarray']:
            for msg_idx in range(len(message)):
                f.write("[{}] = ".format(msg_idx))
                msg = message[msg_idx]
                if type(msg).__name__ in ['list', 'ndarray']:
                    f.write("\n")
                    for m_idx in range(len(msg)):
                        f.write("\t[{}] = ".format(m_idx))
                        m = msg[m_idx]
                        if type(m).__name__ == 'dict':
                            f.write("\n")
                            for key, val in m.items():
                                if key == "ONLY VALUE":
                                    f.write("\t\t{}\n".format(val))
                                else:
                                    f.write("\t\t{} : {}\n".format(key, val))
                        elif (type(m).__name__ in ['list', 'ndarray']) and (len(m) > 0) and (type(m[0]).__name__ in ['list', 'ndarray']):
                            f.write("\n")
                            for idx in range(len(m)):
                                f.write("\t\t[{}] = {}\n".format(idx, m[idx]))
                        else:
                            f.write("\t\t{}\n".format(m))
                elif type(msg).__name__ == 'dict':
                    f.write("\n")
                    for m_key, m in msg.items():
                        if m_key == "ONLY VALUE":
                            f.write("\t{}\n".format(m))
                        else:
                            f.write("\t{} : ".format(m_key))
                            if type(m).__name__ == 'dict':
                                f.write("\n")
                                for key, val in m.items():
                                    if key == "ONLY VALUE":
                                        f.write("\t\t{}\n".format(val))
                                    else:
                                        f.write(
                                            "\t\t{} : {}\n".format(key, val))
                            elif (type(m).__name__ in ['list', 'ndarray']) and (len(m) > 0) and (type(m[0]).__name__ in ['list', 'ndarray']):
                                f.write("\n")
                                for idx in range(len(m)):
                                    f.write("\t\t[{}] = {}\n".format(idx, m[idx]))
                            else:
                                f.write("{}\n".format(m))
                else:
                    f.write("{}\n".format(msg))
        elif type(message).__name__ == 'dict':
            for msg_key, msg in message.items():
                if msg_key == "ONLY VALUE":
                    f.write("{}\n".format(msg))
                else:
                    f.write("{} : ".format(msg_key))
                    if type(msg).__name__ in ['list', 'ndarray']:
                        f.write("\n")
                        for m_idx in range(len(msg)):
                            f.write("\t[{}] = ".format(m_idx))
                            m = msg[m_idx]
                            if type(m).__name__ == 'dict':
                                f.write("\n")
                                for key, val in m.items():
                                    if key == "ONLY VALUE":
                                        f.write("\t\t{}\n".format(val))
                                    else:
                                        f.write(
                                            "\t\t{} : {}\n".format(key, val))
                            elif (type(m).__name__ in ['list', 'ndarray']) and (len(m) > 0) and (type(m[0]).__name__ in ['list', 'ndarray']):
                                f.write("\n")
                                for idx in range(len(m)):
                                    f.write("\t\t[{}] = {}\n".format(idx, m[idx]))
                            else:
                                f.write("\t\t{}\n".format(m))
                    elif type(msg).__name__ == 'dict':
                        f.write("\n")
                        for m_key, m in msg.items():
                            if m_key == "ONLY VALUE":
                                f.write("\t{}\n".format(m))
                            else:
                                f.write("\t{} : ".format(m_key))
                                if type(m).__name__ == 'dict':
                                    f.write("\n")
                                    for key, val in m.items():
                                        if key == "ONLY VALUE":
                                            f.write("\t\t{}\n".format(val))
                                        else:
                                            f.write(
                                                "\t\t{} : {}\n".format(key, val))
                                elif (type(m).__name__ in ['list', 'ndarray']) and (len(m) > 0) and (type(m[0]).__name__ in ['list', 'ndarray']):
                                    f.write("\n")
                                    for idx in range(len(m)):
                                        f.write("\t\t[{}] = {}\n".format(idx, m[idx]))
                                else:
                                    f.write("{}\n".format(m))
                    else:
                        f.write("{}\n".format(msg))
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
    writeLog({
            "lineNum": 47,
            "funName": "testFunc",
            "fileName": "C:/Users/88692/Desktop/Hey-Fish-What-are-you-doing/FishDebug.py"
        },
        "Test_WriteLog",
        {'apple': 1, 'banana': 2}
    )
    print("finish.")


if __name__ == '__main__':
    if len(sys.argv) == 2:
        option = sys.argv[1]
        if option == '-h' or option == '--help':  # see help message
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
