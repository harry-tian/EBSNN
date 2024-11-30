import sys
import os
from shutil import copyfile
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

p_log_file = 'log_20/log_train_flow_K3_gamma1_B128_E64.txt'


def set_log_file(log_filename):
    global p_log_file
    p_log_file = log_filename
    if os.path.exists(p_log_file):
        print('WARNING: file {} already exists, make a backup.'.format(
            p_log_file))
        copyfile(p_log_file, p_log_file + '.bak')
    print('INFO: log will write into {}.'.format(p_log_file))
    p_log_file = open(p_log_file, 'w')


def p_log(*ks, **kwargs):
    print(*ks, **kwargs)
    sys.stdout.flush()
    stdout = sys.stdout
    sys.stdout = p_log_file
    print(*ks, **kwargs)
    sys.stdout.flush()
    sys.stdout = stdout


def deal_results(y_true, y_pred, digits=4):
    p_log('Confison Matrix:\n', confusion_matrix(y_true, y_pred))
    p_log(classification_report(y_true, y_pred, digits=digits))
    return classification_report(y_true, y_pred,
                                 output_dict=True,
                                 digits=digits)


def bytes2bits(x):
    return ''.join(f'{byte:08b}' for byte in x)
def bits2bytes(x):
    return bytes(int(x[i:i+8], 2) for i in range(0, len(x), 8))
def mask(bits, start, end):
    return bits[:start] + '0'*(end-start) + bits[end:]
def bits2ints(b):
    b = b.zfill((len(b) + 7) // 8 * 8)
    return [int(b[i:i+8], 2) for i in range(0, len(b), 8)]

def mask_ip_header(packet):
    from scapy.all import IP
    if IP not in packet:
        return ''
    
    ip_header = bytes(packet[IP])
    ip_header_bits = bytes2bits(ip_header)
    U = int(ip_header_bits[4:8],2)
    ip_header_bits = ip_header_bits[:(U*32)]

    ip_header_bits = mask(ip_header_bits, 32, 48) # identification
    ip_header_bits = mask(ip_header_bits, 80, 96) # checksum
    ip_header_bits = mask(ip_header_bits, 96, 128) # src ip
    ip_header_bits = mask(ip_header_bits, 128, 160) # dst ip
    return bits2bytes(ip_header_bits)

def mask_tcpudp_header(packet):
    from scapy.all import TCP, UDP
    if TCP in packet:
        tcp_len = packet[TCP].dataofs
        header = bytes(packet[TCP])[:(tcp_len*4)]
        header_bits = bytes2bits(header)
    elif UDP in packet:
        header = bytes(packet[UDP])[:8]
        header_bits = bytes2bits(header)
    else:
        return ''

    header_bits = mask(header_bits, 0, 16) # src port
    header_bits = mask(header_bits, 16, 32) # dst port
    return bits2bytes(header_bits)