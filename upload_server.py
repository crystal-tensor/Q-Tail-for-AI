from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import cgi
import json
import subprocess
from werkzeug.utils import secure_filename

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

class UploadHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-type")
        self.end_headers()

    def do_POST(self):
        if self.path == '/upload':
            ctype, pdict = cgi.parse_header(self.headers['content-type'])
            if ctype == 'multipart/form-data':
                content_length = int(self.headers['Content-Length'])
                if content_length > MAX_UPLOAD_SIZE:
                    self.send_response(413)
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(b'Payload Too Large')
                    return

                pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
                pdict['CONTENT-LENGTH'] = content_length
                fields = cgi.parse_multipart(self.rfile, pdict)
                
                # Get the file data
                file_data = fields.get('file')[0]
                raw_file_name = fields.get('filename')[0] if 'filename' in fields else 'uploaded_data.csv'
                
                if isinstance(raw_file_name, bytes):
                    raw_file_name = raw_file_name.decode('utf-8')
                file_name = secure_filename(raw_file_name)
                
                if not file_name.endswith(('.csv', '.json', '.yaml')):
                    self.send_response(400)
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(b'Invalid file type. Only CSV, JSON, and YAML are allowed.')
                    return
                
                # Ensure data directory exists
                os.makedirs('data', exist_ok=True)
                
                # Save the file
                file_path = os.path.join('data', file_name)
                with open(file_path, 'wb') as f:
                    f.write(file_data)
                
                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {'status': 'success', 'message': f'File {file_name} saved successfully to data directory.'}
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_response(400)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(b'Bad Request: Expected multipart/form-data')
        elif self.path == '/generate':
            auth_header = self.headers.get('Authorization')
            if not auth_header or auth_header != 'Bearer SECRET_TOKEN':
                self.send_response(401)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(b'Unauthorized')
                return

            try:
                # Run the real_rcs_pt.py script
                print("Running real_rcs_pt.py...")
                result = subprocess.run(['python3', 'real_rcs_pt.py'], capture_output=True, text=True)
                
                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                if result.returncode == 0:
                    response = {'status': 'success', 'message': '数据生成成功！\n\n' + result.stdout}
                else:
                    response = {'status': 'error', 'message': '脚本执行失败：\n' + result.stderr}
                
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {'status': 'error', 'message': str(e)}
                self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

def run(server_class=HTTPServer, handler_class=UploadHandler, port=8222):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting upload server on port {port}...')
    print(f'Waiting for file uploads to save into ./data/ directory...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()