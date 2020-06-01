import os
import sys
import json
import yaml
import datetime as dt
import pandas as pd


def load_auth_config():
    """
    """
    with open(
        "/Users/peteraltamura/Documents/GitHub/authConfigs/"
        "auth_configuration.yaml", "rb"
    ) as cf:
        CONFIG = yaml.load(cf)

    return CONFIG


def load_config():
    try:
        with open("configuration.yaml", "rb") as ff:
            config = yaml.load(ff)
    except:
        
        with open("../configuration.yaml", "rb") as ff:
            config = yaml.load(ff)

    # Replace Volume if external not mounted
    if not os.path.exists("/Volumes/Samsung_T5/"):
        for k, path_ in config.get('nba').get('paths').items():
            config['nba']['paths'][k] = path_.replace(
                "/Volumes/Samsung_T5",
                ""
            )
    return config


def log_feed_pull_error(fd, error_):

    # Get Error Log Path from Configuration yml
    with open("configuration.yaml", "rb") as ff:
        config = yaml.load(ff)
    log_path = config.get('nba').get('paths').get('error_log')

    # Create error log json
    if str(type(fd)) != "tuple":
        err = {
            'pullDate': str(dt.datetime.now()),
            'errorMessage': str(error_),
            'feed': str(fd),
        }
    else:
        err = {
            'pullDate': str(dt.datetime.now()),
            'errorMessage': str(error_),
            'feed': str(fd[0]),
            'date': str(fd[1])
        }
    log_path += dt.datetime.now().strftime("%Y_%m_%d")
    log_path += "/"

    # Daily Path
    try:
        os.makedirs(log_path)
    except:
        pass

    # Write
    with open(log_path+"{}{}.json".format(fd[0], fd[1]), "w") as js:
        json.dump(err, js)
    print("        {} :: Error Logged".format(fd))


def get_latest_game_version(data):
    assert 'filename' in data.columns
    assert 'gameId' in data.columns
    assert 'gameStartTime' in data.columns
    data.loc[:, 'gameIdTemp'] = \
        data['gameId'].astype(str) + "__" + \
        data['gameStartTime'].astype(str)
    data['maxFilename'] = data.groupby('gameIdTemp')['filename'].\
        transform('max')
    data = data.loc[data['filename'] == data['maxFilename'], :]
    data.drop(labels=['gameIdTemp'], axis=1, inplace=True)
    return data


if __name__ == "__main__":
    pass
