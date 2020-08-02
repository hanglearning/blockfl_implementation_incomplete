import re 
import sys
from flask import Flask, request
import threading
from threading import Event
import requests
import os
import time
import json


import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# function to validate an Ip addess 
def check_ip_format(Ip):  

	ip_regex = '''^(25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.( 
			25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.( 
			25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.( 
			25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)$'''
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
	requiredNamed = parser.add_argument_group('required arguments')
	requiredNamed.add_argument('--port', type=int, help='Specify the port number with the speicified --ip this node will run on. REQUIRED and no default value. Example: 5000', required=True)
	parser.add_argument('--ip', type=str, default='127.0.0.1', help='Specify the IP address this node will run on. Default on localhost. Example: 127.0.0.1')
	parser.add_argument('--registrar_ip_port', type=str, default='', help="Specify the peer's IP address with the port number as a registrar for this node to register in the existing network. Example: 127.0.0.1:5001\n NOTE: This argument should be skipped if this node is the first node running in the network.")
	# add -d to specify device, miner or worker in the final version
	args = parser.parse_args()
	return check_input_args(args)

def check_input_args(args):
	if not check_ip_format(args.ip):
		sys.exit("Invalid IP address specified. Example --ip 127.0.0.1")
	if not 1 <= args.port <= 65535:
		sys.exit("Invalid port number specified. It must be in the range [1, 65535]. Example: --port 5001")
	if args.registrar_ip_port:
		registrar_ip_port_list = args.registrar_ip_port.split(':')
		if not check_ip_format(registrar_ip_port_list[0]):
			sys.exit("Invalid registrar IP address specified. Example: --registrar_ip_port 127.0.0.1:5001")
		if not 1 <= int(registrar_ip_port_list[1]) <= 65535:
			sys.exit("Invalid registrar port number specified. It must be in the range [1, 65535]. Example: --registrar_ip_port 127.0.0.1:5001")
	return args.ip, args.port, args.registrar_ip_port

app = Flask(__name__)

def run_flask_node(ip, port):
	app.run(host=ip, port=port)

def check_start_app(this_node_address, registrar_ip_port):
	retry_registration = 3
	retry_registration_count = retry_registration
	retry_wait_time = 5
	while True:
		try:
			response = requests.get(f'{this_node_address}/running')
			while response.status_code != 888:
				response = requests.get(f'{this_node_address}/running')
			# register in the network
			if registrar_ip_port:
				registrar_ip_port = f"http://{registrar_ip_port}"
				data = {"registrar_node_address": registrar_ip_port}
				headers = {'Content-Type': "application/json"}
				try:
					response_register = requests.post(f"{this_node_address}/register_with", data=json.dumps(data), headers=headers)
				except Exception as e: 
					print(e)
					if retry_registration_count:
						print(f"Cannot start the registration from this node. {retry_registration_count} re-attempts left...")
						retry_registration_count -= 1
						print(f"Reconnecting in {retry_wait_time} seconds...")
						time.sleep(retry_wait_time)
						continue
					else:
						print(f"Cannot start the registration from this node after {retry_registration} attempts. Please check the port number, restart the app and try again.")
						os._exit(0)
				if response_register.status_code != 201:
					print(f"Registration error. Response code {response_register.status_code} with {response_register.text}. Please restart the app and try again.")
					os._exit(0)
			else:
				print("No registrar address specified. This node will run as a registrar.")
			print(f"Node is ready serving at {this_node_address}")
			break
		except:
			continue

def start_flask_app(ip, port, this_node_address, registrar_ip_port):
	t1 = threading.Thread(target = run_flask_node, args=(ip, port))
	t2 = threading.Thread(target = check_start_app, args=(this_node_address, registrar_ip_port))
	t1.start()
	t2.start()