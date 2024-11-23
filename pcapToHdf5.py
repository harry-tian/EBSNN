import os
import sys
import h5py
from scapy.all import rdpcap

def pcap_to_hdf5(pcap_file, hdf5_file):
    try:
        packets = rdpcap(pcap_file)
    except Exception as e:
        print(f"Error reading PCAP file: {e}")
        return

    try:
        with h5py.File(hdf5_file, 'w') as hdf5:

            dataset = hdf5.create_dataset(
                'packets', 
                (len(packets),), 
                dtype=h5py.string_dtype(encoding='utf-8')
            )

            for i, packet in enumerate(packets):
                dataset[i] = str(packet) 
                
        print(f"Converted {pcap_file} to {hdf5_file} successfully.")
    except Exception as e:
        print(f"Error writing HDF5 file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <pcap_file>")
        sys.exit(1)
    
    pcap_file = sys.argv[1]
    if not os.path.exists(pcap_file):
        print(f"PCAP file {pcap_file} does not exist.")
        sys.exit(1)

    hdf5_file = os.path.splitext(pcap_file)[0] + '.hdf5'

    pcap_to_hdf5(pcap_file, hdf5_file)
