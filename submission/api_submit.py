# coding: utf-8
import requests

files={'files': open('sample_submission.csv','rb')}

data = {
    "user_id": "zhouli",   #user_id is your username which can be found on the top-right corner on our website when you logged in.
    "team_token": "cf126c79d6354649909de480580691ee3219c9feddf95086a8caaefc7a1a1269", #your team_token.
    "description": '1',  #no more than 40 chars.
    "filename": "random", #your filename
}

url = 'https://biendata.com/competition/kdd_2018_submit/'

response = requests.post(url, files=files, data=data)

print(response.text)


