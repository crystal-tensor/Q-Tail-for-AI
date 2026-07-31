import os
import requests
from quafu.users.userapi import User
token = os.environ.get("QUAFU_TOKEN")
if not token:
    raise ValueError("Please set QUAFU_TOKEN environment variable.")
print(f"Token: {token[:10]}...{token[-10:]}")
user = User(api_token=token)
user.url = "https://quafu.baqis.ac.cn/"
try:
    print(user.get_available_backends(print_info=False))
except Exception as e:
    print(f"Error: {e}")
