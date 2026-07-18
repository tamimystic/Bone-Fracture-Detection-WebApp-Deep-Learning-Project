import traceback
import sys

def simulate_error():
    pass

try:
    simulate_error()
except Exception as e:
    print(traceback.format_exc())
