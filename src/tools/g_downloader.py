import requests
import argparse
import sys

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)




def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--FILE_ID', type=str,
                        help='File id')

    parser.add_argument('--OUTPUT_PATH', type=str,
                        help='Path where output of module will be stored')

    return parser.parse_args(argv)

def main(cfg):
    file_id = cfg.FILE_ID
    destination = cfg.OUTPUT_PATH
    download_file_from_google_drive(file_id, destination)

if __name__ == '__main__':
    cfg = parse_arguments(sys.argv[1:])
    main(cfg)
