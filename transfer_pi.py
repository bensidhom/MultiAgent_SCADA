import paramiko
import os

# SSH and file transfer settings
host = "150.250.210.249"
username = "pi"
password = "1234"
local_path = r"inferece_deployement\spotwave_ae.py"  # Ensure this file is in the same folder as this script
remote_path = "/home/pi/spotwave_ae.py"

def upload_file():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        print("Connecting to Raspberry Pi...")
        ssh.connect(hostname=host, username=username, password=password)

        sftp = ssh.open_sftp()
        print(f"Transferring {local_path} to {remote_path}...")
        sftp.put(local_path, remote_path)
        sftp.chmod(remote_path, 0o755)  # Make it executable just in case

        print("File transfer complete.")
        sftp.close()
    except Exception as e:
        print(f"File transfer failed: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    upload_file()
