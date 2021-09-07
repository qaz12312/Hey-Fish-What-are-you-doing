# log format
import logging
import os
import sys
import traceback
import datetime
import json

# 設定目錄、檔名
dir_path = os.path.dirname(os.path.abspath(__file__)) + '/log/'
file_path = dir_path + "{:%Y-%m-%d}".format(datetime.datetime.now()) + '.log'

# 若不存在則新建
if not os.path.isdir(dir_path):
    os.makedirs(dir_path)
if not os.path.isfile(file_path):
    with open(file_path, 'w'): pass


logging.basicConfig(
    level = logging.DEBUG,
    filename = file_path, 
    filemode = 'a+',
    format = '[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S',
)

# -----------------------------------------------------------
# Write a log
# -----------------------------------------------------------
def writeLog(info):
    level = info['logLevel']
    log = json.dumps(info['message'])
    # write log
    if level == logging.CRITICAL:
        logging.critical(log)
    elif level == logging.ERROR:
        logging.error(log)
    elif level == logging.WARNING:
        logging.warning(log)
    elif level== logging.INFO:
        logging.info(log)
    else:
        logging.debug(log)
    # ---------------------------------------------------------------------
    # way 2 :
    # # config
    # logging.captureWarnings(True)# 捕捉 py waring message
    # formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    # my_logger = logging.getLogger('py.warnings')# 捕捉 py waring message
    # my_logger.setLevel(logging.INFO)

    # # file handler
    # fileHandler = logging.FileHandler(file_path, 'w', 'utf-8')
    # fileHandler.setFormatter(formatter)
    # my_logger.addHandler(fileHandler)
    # # console handler - 也在 console 出現
    # consoleHandler = logging.StreamHandler()
    # consoleHandler.setLevel(logging.DEBUG)
    # consoleHandler.setFormatter(formatter)
    # my_logger.addHandler(consoleHandler)
    # ---------------------------------------------------------------------


# -----------------------------------------------------------
# Format a log
# -----------------------------------------------------------
def formatLog(level, programPath, programName, action, message = None):
    log_info = {
        'logLevel': level,
        'message' : {
            'state' : "Success",
            'action'     : action,
            'message'    :'',
            'programPath': programPath,
            'programName': programName,
        }
    }
    if message != None:
        log_info['message']['message'] = message
    return log_info


# -----------------------------------------------------------
# Format an exception
# -----------------------------------------------------------
def formatException(e_msg, programInfo, action, message = None):
    if programInfo[2]=='<module>':
        programName = "line {}".format(programInfo[1])
    else :
        programName = "line {}, in {}()".format(programInfo[1],programInfo[2])
    
    log_info = {
        'logLevel': logging.CRITICAL,
        'message' : {
            'state' : "Failure",
            'action'     : action,
            'message'    : e_msg,
            'programPath': programInfo[0],
            'programName': programName,
        }
    }
    if message != None:
        log_info['message']['message'] += message
    return log_info


# -----------------------------------------------------------
# Get FishLog.py info
# -----------------------------------------------------------
def help():
    msg = """
    Log Level:
        CRITICAL: 50 (FATAL = CRITICAL)
        ERROR   : 40
        WARNING : 30 (WARN = WARNING)
        INFO    : 20
        DEBUG   : 10

    formatLog() args:
        level         : int,    Log Level
        programPath   : string, The absolute path of the file
        programName   : string, The line where the function is called and function Name ("line {lineNum}, in {funName}")
        action        : string, Action performed
        message = None: string, More message
    
    formatException() args:
        e_msg         : string, message of Exception (e.args[0])
        programInfo   : dict,   traceback info (traceback.extract_tb(tb)[-1] (cl, exc, tb = sys.exc_info()))
        action        : string, Action performed
        message = None: string, More message
    
    writeLog() args:
        info          : dict, is return value of formatLog() or formatException()
    """
    print(msg)


# -----------------------------------------------------------
# Test for each function
# -----------------------------------------------------------
def testFunc():
    print("test formatLog().")
    log_info = formatLog(10, "Hey-Fish-What-are-you-doing\FishLog.py", "line 142, in testFunc", "Test formatLog()")
    writeLog(log_info)
    print("finish writeLog().")

    print("test formatException().")
    try:
        raise ValueError('Test.')
    except Exception as e:
        cl, exc, tb = sys.exc_info() #取得Call Stack
        lastCallStack = traceback.extract_tb(tb)[-1] #取得Call Stack的最後一筆資料
        log_info = formatException(e.args[0], lastCallStack, "Test formatException()")
        writeLog(log_info)
    finally:
        print("finish writeLog().")


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
    FishLog.py [options]

    Options:
    -h, --help  Display function args info
        --test  Run test for each function
    """
    print(print_help)