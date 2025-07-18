from flask import Flask, request, jsonify
import os
from datetime import datetime
import socket

app = Flask(__name__)
UPLOAD_FOLDER = 'received_photos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


@app.route('/')
def test_connection():
    return jsonify({
        'status': 'success',
        'message': 'Server is ready',
        'server_ip': get_local_ip(),
        'client_ip': request.remote_addr
    }), 200


@app.route('/upload', methods=['POST'])
def upload_file():
    client_ip = request.remote_addr
    print(f"\n--- New upload request from {client_ip} ---")

    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({'status': 'error', 'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        print("Empty filename")
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    if file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{timestamp}.jpg"
        save_path = os.path.join(UPLOAD_FOLDER, filename)

        try:
            file.save(save_path)
            file_size = os.path.getsize(save_path)
            print(f"Saved: {filename} ({file_size} bytes) from {client_ip}")
            return jsonify({'status': 'success', 'filename': filename}), 200
        except Exception as e:
            print(f"Save error: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    host_ip = get_local_ip()
    print(f"\n🔥 Starting server at http://{host_ip}:8080")
    print("Available endpoints:")
    print(f"  • GET /       - Server test")
    print(f"  • POST /upload - Upload endpoint\n")

    app.run(host='0.0.0.0', port=8080, debug=True)


# WIB + R - cmd - ipconfig
# Запускать следующую команду через PowerShell от имени администратора
# New-NetFirewallRule -DisplayName "Allow Port 8080" -Direction Inbound -LocalPort 8080 -Protocol TCP -Action Allow
