# air_quality_prediction

## 在项目目录下建立data子目录，并建立如下的目录级别
  <!-- TOC -->

- [data](#air_quality_prediction)
  - [Beijing](#air_quality_prediction)
    - [aq](#air_quality_prediction)
    - [grid_meo](#air_quality_prediction)
    - [location](#air_quality_prediction)
  - [London](#air_quality_prediction)
    - [aq](#air_quality_prediction)
    - [grid_meo](#air_quality_prediction)
    - [location](#air_quality_prediction)
  - [Newdata](#air_quality_prediction)
    - [Beijing](#air_quality_prediction)
      - [aq](#air_quality_prediction)
      - [grid_meo](#air_quality_prediction)
      - [location](#air_quality_prediction)
    - [London](#air_quality_prediction)
      - [aq](#air_quality_prediction)
      - [grid_meo](#air_quality_prediction)
      - [location](#air_quality_prediction)
  - [Newdata_processed](#air_quality_prediction)
  - [processed_data](#air_quality_prediction)


<!-- /TOC -->
## 配置定时任务的方案
1. 将本项目模块加入python系统库目录
   
   1.1 创建文件 air_quality_prediction.pth
   
   1.2 写入/home/competition/air_quality_prediction（项目的根目录）
   
   1.3 将文件移动到python的site-packages目录下面

2. 配置crontab定时任务

   2.1 在shell里执行 crontab -e
   
   2.2 输入 22 23 * 5  *  py3 /home/competition/air_quality_prediction/utils/crontab.py    
   
   上面的输入含义是：5月份的每天23点22分执行命令 py3 /home/competition/air_quality_prediction/utils/crontab.py 
   此处的py3应写为本地的python