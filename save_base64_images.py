import base64
import os

os.makedirs("results", exist_ok=True)

img1_b64 = """
iVBORw0KGgoAAAANSUhEUgAAA2AAAAJHCAIAAACgD76vAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUA
...
""" # Need the actual base64 content from the user's message
# I will use a simple script to read the images provided by the user directly if possible, or I will ask the user to upload them as files.
