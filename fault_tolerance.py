# FAULT TOLERANCE
RETRY_CONNECTION_TIMES = 3 #TODO for every response
RETRY_ERROR_RESPONSE = 3 #TODO for every response status code
RETRY_WAITING_TIME_IN_SECONDS = 5

def retry_connection(end_point, success_status_code, error_msg, retry_times=RETRY_CONNECTION_TIMES, waiting_time=RETRY_WAITING_TIME_IN_SECONDS):
    pass

def retry_error_response():
    pass