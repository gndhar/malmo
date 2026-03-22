import socket
import threading
import time
import struct

# Constants from assignment requirements
SERVER_IP = "127.0.0.1"
SERVER_PORT = 12345
TOTAL_PACKETS = 10000
TIMEOUT_INTERVAL = 0.5  # Adjust based on RTT scenarios in Table 1


class GBNClient:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.1)  # Short timeout for non-blocking recv

        self.base = -1  # Last contiguous ACKed packet
        self.next_seq_num = 0  # Next packet to send
        self.cwnd = 1.0  # AIMD Congestion Window
        self.last_ack_time = time.time()

        self.lock = threading.Lock()
        self.running = True

    def receiver(self):
        """Listens for cumulative ACKs and updates window state."""
        while self.running:
            try:
                data, _ = self.sock.recvfrom(1024)
                ack_num = struct.unpack("!I", data)[0]  # Big-endian 4-byte

                with self.lock:
                    if ack_num > self.base:
                        # Additive Increase: CWND += 1 per window
                        self.cwnd += 1.0 / int(self.cwnd)
                        self.base = ack_num
                        self.last_ack_time = time.time()  # Reset timer on progress
                        # print(f"ACK {ack_num} | CWND: {self.cwnd:.2f}")
            except socket.timeout:
                continue

    def sender(self):
        """Manages the window and handles GBN retransmissions."""
        while self.base < TOTAL_PACKETS - 1:
            with self.lock:
                # 1. Check for Timeout (Loss Detection) [cite: 30]
                if time.time() - self.last_ack_time > TIMEOUT_INTERVAL:
                    print(f"Timeout! Resetting to {self.base + 1}")
                    # Multiplicative Decrease
                    self.cwnd = max(1.0, self.cwnd / 2.0)
                    # Go-Back-N: Start over from the oldest un-ACKed packet
                    self.next_seq_num = self.base + 1
                    self.last_ack_time = time.time()

                # 2. Send Packets within the CWND [cite: 33]
                while (
                    self.next_seq_num < TOTAL_PACKETS
                    and self.next_seq_num <= self.base + int(self.cwnd)
                ):

                    packet = struct.pack("!I", self.next_seq_num)
                    self.sock.sendto(packet, (SERVER_IP, SERVER_PORT))
                    self.next_seq_num += 1

            # Small yield to prevent CPU pinning
            time.sleep(0.001)

    def run(self):
        start_time = time.time()
        threading.Thread(target=self.receiver, daemon=True).start()

        self.sender()

        duration = time.time() - start_time
        print(f"\nTransfer Complete.")
        print(f"Total Time: {duration:.2f}s")
        print(f"Throughput: {TOTAL_PACKETS / duration:.2f} PPS")


if __name__ == "__main__":
    client = GBNClient()
    client.run()
