import os
import time
import datetime
import logging
import requests
from utils.config import KddConfig

logfilepath = KddConfig.task_log_file_path+"runlog"+datetime.datetime.now().strftime('%m_%d_%H_%M_%S')#"/home/fan/runlog"
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',filename=logfilepath, filemode='a')
logger = logging.getLogger(__name__)

def submit(user_id, description, filename):
    submission_file = KddConfig.submission_formal_file_path
    files = {'files': open(submission_file, 'rb')}
    data = {
        "user_id": user_id,#"JohnLee",
        # user_id is your username which can be found on the top-right corner on our website when you logged in.
        "team_token": "cf126c79d6354649909de480580691ee3219c9feddf95086a8caaefc7a1a1269",  # your team_token.
        "description": description,#'beijing200-90-l2loss-london-200-195huberloss-23hour',  # no more than 40 chars.
        "filename": filename #"201805-5",  # your filename
    }

    url = 'https://biendata.com/competition/kdd_2018_submit/'

    response = requests.post(url, files=files, data=data)
    return response.text

def task():
    # download data
    retryCount = 1
    retryUplimit = 10
    while (retryCount <= retryUplimit):
        project_dir = KddConfig.project_dir
        cmd = KddConfig.python_path+" "+KddConfig.project_dir+"utils/get_data_from_start_to_end.py"
        downloadResult = os.system(cmd)
        if downloadResult != 0:
            logger.info("download error")
            time.sleep(60)
            retryCount = retryCount + 1
        else:
            break
    if retryCount > retryUplimit:
        logger.error("download error")
        return

    # generate dev_data
    cmd = KddConfig.python_path+" "+KddConfig.project_dir+"train/train_data_generate.py"
    genDevRes = os.system(cmd)
    if genDevRes != 0:
        logger.error("error generate dev_data")
        return

    # use new data to predict
    cmd = KddConfig.python_path + " " + KddConfig.project_dir + "eval/eval_seq2seq.py"
    predictResult = os.system(cmd)
    if predictResult != 0:
        logger.error("error eval")
        return

    # check negative value
    now = datetime.datetime.utcnow()
    logger.info("filename"+now.strftime('%Y-%m-%d %H:%M:%S'))
    # sumbit
    submitResult = submit("JohnLee","beijing200-90-l2loss-london-200-195huberloss-23hour", "filename"+now.strftime('%Y-%m-%d %H:%M:%S'))
    if "true" in submitResult:
        logger.info("submit success")
    else:
        logger.error(submitResult)

if __name__   == '__main__':
    task()
