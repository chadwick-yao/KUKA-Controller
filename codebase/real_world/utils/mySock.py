import socket
import time
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class mySock(object):

    def __init__(self,
                 remote_host: str,
                 remote_port: int,
                 trans: Tuple=(0,0,0,0,0,0),
                 **kwargs
                 ) -> None:
        assert isinstance(trans, tuple) and len(trans) == 6, "TCP transform shall be a tuple of 6."

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            self.sock.connect((remote_host, remote_port))
            logger.info(f"Remote device was connected. (IP: {remote_host}, PORT: {remote_port})")
            time.sleep(1)
        except:
            self.sock.close()
            raise ConnectionRefusedError("Connection destroyed...")
        
        # Update the transform of the TCP if one is specified
        if all(num == 0 for num in trans):
            print('No TCP transform in Flange Frame is defined.')
            print(f'The following (default) TCP transform is utilized: {trans}')
            return

        print('Trying to mount the following TCP transform:')
        string_tuple = ('x (mm)', 'y (mm)', 'z (mm)', 'alfa (rad)', 'beta (rad)', 'gamma (rad)')
        
        for i in range(6):
            print(string_tuple[i] + ': ' + str(trans[i]))
        
        da_message = 'TFtrans_' + '_'.join(map(str, trans)) + '\n'
        
        try:
            self.send(da_message)
            return_ack_nack = self.receive()
        except:
            raise RuntimeError("Could not mount the specified TCP")
        
        if 'done' in return_ack_nack:
            logger.info('Specified TCP transform mounted successfully')
        else:
            raise RuntimeError("Could not mount the specified TCP")
        
    def send(self, msg: str):
        self.sock.send(msg.encode("utf-8"))

    def receive(self):
        return self.sock.recv(1024).decode("utf-8")
    
    def close(self):
        self.send("end\n")
        time.sleep(1)
        self.sock.close()