import requests
import configparser
config = configparser.ConfigParser()
config.read_file(open('privateconfig'))
token=config['Notification']['token']

def notify(msg='plots ready', group='lab',title='plot'):
    notification="https://api.day.app/{}/{}/{}?group={}".format(token,title, msg, group)
    requests.get(notification)


