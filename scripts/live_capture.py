#!/usr/bin/env python3
"""
live_capture.py

Answers A2 feedback gap #1: "the proposal does not explain how live
network packets on the Raspberry Pi will actually be captured and
converted into the six UNSW-NB15 features required by the trained
model."

This script:
  1. Sniffs live packets on a network interface (via scapy).
  2. Groups packets into bidirectional flows, keyed by the 5-tuple
     (protocol, source IP, source port, destination IP, destination port),
     matching UNSW-NB15's flow definition.
  3. Tracks running totals per flow (bytes in each direction, packet
     timestamps, TCP flags seen).
  4. When a flow completes (FIN/RST seen, or an idle timeout elapses),
     derives the 6 trained features from the accumulated flow data:
         dur     - flow duration in seconds (last packet time - first)
         proto   - transport protocol (tcp/udp/icmp/...), integer-encoded
         sbytes  - total bytes from the flow's initiator
         dbytes  - total bytes from the flow's responder
         state   - simplified connection-state classification (see
                   LIMITATIONS below)
         service - derived from the destination port (e.g. 80 -> http)
  5. Encodes proto/state/service using category_mappings.json (the
     exact mapping recovered from the original training data - see
     recover_original_mapping.py), so live traffic is encoded
     identically to how the model was trained.
  6. Feeds the resulting feature vector into the same trained model
     used by run_inference.py, via ImpulseRunner.classify().

LIMITATIONS (documented deliberately, for the A3 risks section):
  - `state` in the real UNSW-NB15 dataset was derived using Bro/Zeek's
    full TCP state-machine tracking, which models the complete
    connection lifecycle (handshake negotiation, retransmissions,
    simultaneous close, etc.). This script implements a SIMPLIFIED
    heuristic (see `derive_state()` below) based on which TCP flags
    were observed and packet count - it will not perfectly match
    Zeek's classification for edge cases, but captures the dominant
    states (FIN, RST, CON, INT) correctly for typical flows.
  - `service` is derived purely from destination port number. This
    misses non-standard port usage (e.g. HTTP served on a non-80 port)
    - the original dataset likely used deeper protocol inspection.
  - Flow completion uses a fixed idle timeout (default 30s) for
    non-terminated flows (e.g. UDP has no FIN/RST), which is a
    reasonable but simplified approximation of flow boundaries.

Usage (on the Raspberry Pi, requires root for packet capture):
    sudo python3 live_capture.py --interface eth0

    sudo python3 live_capture.py --interface eth0 --idle-timeout 30
"""

import argparse
import json
import os
import threading
import time

from scapy.all import sniff, IP, TCP, UDP, ICMP

from edge_impulse_linux.runner import ImpulseRunner


MAPPING_PATH = '/home/pi/category_mappings.json'
MODEL_PATH = '/home/pi/unsw-nb15-network-anomaly-detection-linux-armv7-v3-impulse-eon-9e6.eim'

# Destination-port -> service name lookup. Names match the exact
# strings present in category_mappings.json (recovered from training
# data), so encoding stays consistent with the trained model.
PORT_SERVICE_MAP = {
    20: 'ftp-data', 21: 'ftp', 22: 'ssh', 25: 'smtp',
    53: 'dns', 67: 'dhcp', 68: 'dhcp', 80: 'http',
    110: 'pop3', 161: 'snmp', 162: 'snmp',
    194: 'irc', 6667: 'irc',
    443: 'ssl', 1812: 'radius', 1813: 'radius',
}

# scapy IP.proto numeric values -> protocol name strings, matching
# category_mappings.json's proto keys.
IP_PROTO_NUM_TO_NAME = {
    1: 'icmp', 6: 'tcp', 17: 'udp', 2: 'igmp', 89: 'ospf',
    47: 'gre', 50: 'esp', 51: 'ah', 41: 'ipv6',
}


def load_mappings():
    with open(MAPPING_PATH, 'r') as f:
        return json.load(f)


def encode(value, mapping, field_name):
    """Encode a category string using the recovered training mapping.
    Falls back to -1 (and prints a warning) for any category never
    seen during training - this is a real limitation worth flagging:
    live traffic can contain protocols/services the training data
    never had examples of."""
    if value in mapping:
        return mapping[value]
    print(f"  WARNING: '{value}' not in trained {field_name} categories "
          f"(never seen in training data) - encoding as -1")
    return -1


def derive_service(dport):
    return PORT_SERVICE_MAP.get(dport, '-')


def derive_proto_name(pkt):
    if pkt.haslayer(IP):
        proto_num = pkt[IP].proto
        return IP_PROTO_NUM_TO_NAME.get(proto_num, None)
    return None


class Flow:
    """Tracks one bidirectional network flow from first packet to
    completion. `orig_*` fields refer to whichever side sent the
    first packet (matching UNSW-NB15's srcip/sbytes convention)."""

    def __init__(self, proto_name, orig_ip, orig_port, resp_ip, resp_port,
                 first_ts):
        self.proto_name = proto_name
        self.orig_ip = orig_ip
        self.orig_port = orig_port
        self.resp_ip = resp_ip
        self.resp_port = resp_port
        self.first_ts = first_ts
        self.last_ts = first_ts
        self.sbytes = 0   # bytes sent by the originator
        self.dbytes = 0   # bytes sent by the responder
        self.tcp_flags_seen = set()
        self.packet_count = 0

    def add_packet(self, pkt, ts, direction):
        size = len(pkt)
        if direction == 'orig_to_resp':
            self.sbytes += size
        else:
            self.dbytes += size
        self.last_ts = ts
        self.packet_count += 1
        if pkt.haslayer(TCP):
            flags = pkt[TCP].flags
            if flags & 0x01:
                self.tcp_flags_seen.add('FIN')
            if flags & 0x04:
                self.tcp_flags_seen.add('RST')
            if flags & 0x02:
                self.tcp_flags_seen.add('SYN')

    def is_terminated(self):
        return 'FIN' in self.tcp_flags_seen or 'RST' in self.tcp_flags_seen

    def derive_state(self):
        """Simplified state classification - see module docstring
        LIMITATIONS for how this differs from the original Bro/Zeek
        derivation used in the UNSW-NB15 dataset."""
        if 'RST' in self.tcp_flags_seen:
            return 'RST'
        if 'FIN' in self.tcp_flags_seen:
            return 'FIN'
        if self.proto_name == 'icmp':
            return 'ECO'
        if 'SYN' in self.tcp_flags_seen and self.packet_count <= 2:
            return 'REQ'
        if self.packet_count <= 1:
            return 'INT'
        return 'CON'

    def to_feature_vector(self, mappings):
        dur = max(0.0, self.last_ts - self.first_ts)
        proto_code = encode(self.proto_name, mappings['proto'], 'proto')
        state_name = self.derive_state()
        state_code = encode(state_name, mappings['state'], 'state')
        service_name = derive_service(self.resp_port) if self.resp_port else '-'
        service_code = encode(service_name, mappings['service'], 'service')
        return [dur, proto_code, self.sbytes, self.dbytes, state_code, service_code]


class FlowTracker:
    def __init__(self, mappings, runner, idle_timeout=30):
        self.mappings = mappings
        self.runner = runner
        self.idle_timeout = idle_timeout
        self.flows = {}       # key -> Flow
        self.lock = threading.Lock()
        self._stop = False

    def _flow_key(self, proto_name, ip_a, port_a, ip_b, port_b):
        """Returns (key, direction) - key is direction-independent so
        both directions of one conversation map to the same flow;
        direction tells us whether this packet is orig->resp or
        resp->orig relative to whichever side we saw first."""
        forward = (proto_name, ip_a, port_a, ip_b, port_b)
        reverse = (proto_name, ip_b, port_b, ip_a, port_a)
        with self.lock:
            if forward in self.flows:
                return forward, 'orig_to_resp'
            if reverse in self.flows:
                return reverse, 'resp_to_orig'
        return forward, 'orig_to_resp'  # new flow, ip_a is the originator

    def handle_packet(self, pkt):
        if not pkt.haslayer(IP):
            return
        proto_name = derive_proto_name(pkt)
        if proto_name is None:
            return  # unhandled protocol, skip

        ip = pkt[IP]
        sport = dport = None
        if pkt.haslayer(TCP):
            sport, dport = pkt[TCP].sport, pkt[TCP].dport
        elif pkt.haslayer(UDP):
            sport, dport = pkt[UDP].sport, pkt[UDP].dport

        ts = time.time()
        key, direction = self._flow_key(proto_name, ip.src, sport, ip.dst, dport)

        with self.lock:
            if key not in self.flows:
                self.flows[key] = Flow(proto_name, ip.src, sport, ip.dst, dport, ts)
            flow = self.flows[key]
            flow.add_packet(pkt, ts, direction)
            terminated = flow.is_terminated()

        if terminated:
            self._finalize(key)

    def sweep_idle_flows(self):
        """Call periodically (e.g. every few seconds) to finalize
        flows that have gone quiet without a clean FIN/RST - mainly
        matters for UDP and ICMP, which have no termination signal."""
        now = time.time()
        with self.lock:
            idle_keys = [
                k for k, f in self.flows.items()
                if now - f.last_ts > self.idle_timeout
            ]
        for key in idle_keys:
            self._finalize(key)

    def _finalize(self, key):
        with self.lock:
            flow = self.flows.pop(key, None)
        if flow is None:
            return
        features = flow.to_feature_vector(self.mappings)
        try:
            result = self.runner.classify(features)
            predicted = result.get('result', {}).get('classification', {})
            top_label = max(predicted, key=predicted.get) if predicted else 'unknown'
            print(f"Flow {flow.orig_ip}:{flow.orig_port} -> "
                  f"{flow.resp_ip}:{flow.resp_port} [{flow.proto_name}] "
                  f"dur={features[0]:.3f}s sbytes={features[2]} dbytes={features[3]} "
                  f"state={flow.derive_state()} service={derive_service(flow.resp_port)} "
                  f"=> {top_label} ({predicted})")
        except Exception as e:
            print(f"  ERROR classifying flow {key}: {e}")

    def idle_sweep_loop(self, interval=5):
        while not self._stop:
            time.sleep(interval)
            self.sweep_idle_flows()

    def stop(self):
        self._stop = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface', default='eth0',
                         help='Network interface to capture on (default: eth0)')
    parser.add_argument('--idle-timeout', type=float, default=30,
                         help='Seconds of inactivity before finalizing a '
                              'non-terminated flow (default: 30)')
    args = parser.parse_args()

    print("=" * 50)
    print("UNSW-NB15 Live Network Traffic Classifier")
    print(f"Interface: {args.interface} | Idle timeout: {args.idle_timeout}s")
    print("=" * 50)

    mappings = load_mappings()
    print(f"Loaded category mappings: "
          f"{len(mappings['proto'])} proto, "
          f"{len(mappings['state'])} state, "
          f"{len(mappings['service'])} service categories")

    runner = ImpulseRunner(MODEL_PATH)
    model_info = runner.init()
    print("Labels: " + str(model_info['model_parameters']['labels']))
    print()

    tracker = FlowTracker(mappings, runner, idle_timeout=args.idle_timeout)

    sweep_thread = threading.Thread(target=tracker.idle_sweep_loop, daemon=True)
    sweep_thread.start()

    print(f"Capturing on {args.interface}... (Ctrl+C to stop)\n")
    try:
        sniff(iface=args.interface, prn=tracker.handle_packet, store=False)
    except KeyboardInterrupt:
        print("\nStopping capture...")
    finally:
        tracker.stop()
        runner.stop()


if __name__ == '__main__':
    main()