import re 
import sys
from flask import Flask, request
import requests

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# NOT USED
def check_candidate_node_address(input_address):
    # https://stackoverflow.com/questions/7160737/python-how-to-validate-a-url-in-python-malformed-or-not
    import re
    regex = re.compile(
            r'^(?:http|ftp)s?://' # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
            r'localhost|' #localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
            r'(?::\d+)?' # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, input_address) is not None

# check ip address
# https://www.geeksforgeeks.org/python-program-to-validate-an-ip-address/
# Make a regular expression 
# for validating an Ip-address 
ip_regex = '''^(25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.( 
            25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.( 
            25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.( 
            25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)$'''
      
# Define a function for 
# validate an Ip addess 
def check_ip_format(Ip):  
    # pass the regular expression 
    # and the string in search() method 
    if (re.search(ip_regex, Ip)) or Ip == '':  
        return True
    else:  
        return False

# https://github.com/hanglearning/ucb-pacman-multiagent-reinforcement/blob/master/pacman.py
# https://www.youtube.com/watch?v=Y4thgo1MUU0
def parse_commands():
    import argparse
    usageStr = """
    BlockFL demo CLI input arguments help.
    """
    parser = argparse.ArgumentParser(usageStr)
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='To specify the IP address this node will run on. Default on localhost. Example - 127.0.0.1')
    parser.add_argument('--registrar_ip_port', type=str, default='', help="To specify the peer's IP address with the port number as a registrar for this node to register in the existing network. Example: 127.0.0.1:5001. NOTE: This argument should be skipped if this node is the first node running in the network.")
    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument('--port', type=int, help='To specify the port number assigned with the speicified IP address this node will run on. Required and no default value. Example - 5000.', required=True)
    # add -d to specify device, miner or worker in the final version
    args = parser.parse_args()
    return check_input_args(args)

def check_input_args(args):
    if not check_ip_format(args.ip):
        sys.exit("Invalid IP address specified. Example --ip 127.0.0.1")
    if not 1 <= args.port <= 65535:
        sys.exit("Invalid port number specified. It must be in [1, 65535]. Example --registrar_ip_port 127.0.0.1:5001")
    if not args.registrar_ip_port:
        registrar_ip_port_list = args.registrar_ip_port.split(':')
        if not check_ip_format(registrar_ip_port_list[0]):
            sys.exit("Invalid registrar IP address specified. Example --registrar_ip_port 127.0.0.1:5001")
        if not 1 <= int(registrar_ip_port_list[1]) <= 65535
            sys.exit("Invalid registrar port number specified. It must be in [1, 65535]. Example --registrar_ip_port 127.0.0.1:5001")
    return args.ip, args.port, args.registrar_ip_port

registrar_ip_port = None

def run_flask_node():
    global registrar_ip_port
    ip, port, registrar_ip_port = parse_commands()
    app.run(host=ip, port=port)


def check_start_app():
	while True:
		try:
			response = requests.get(f'http://{host}:{port}')
			while response.status_code != 200:
				response = requests.get(f'http://{host}:{port}')
			# register peer
			break
		except:
			continue

t1 = threading.Thread(target = run_flask_node)
t2 = threading.Thread(target = check_start_app)

def main():
    t1.start()
    t2.start()