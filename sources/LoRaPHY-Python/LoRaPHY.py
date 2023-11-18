# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 10:17:33 2023

@author: hkl
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.signal import resample
#import galois.GF as gf
#import crcmod.predefined
from fastcrc import crc16


def lowpass_filter(sig, cutoff_freq, fs):
    # Define the filter order (you may need to adjust this)
    filter_order = 4

    # Normalize the cutoff frequency
    normalized_cutoff_freq = cutoff_freq / (0.5 * fs)

    # Create a low-pass Butterworth filter
    b, a = butter(filter_order, normalized_cutoff_freq, btype='low')

    # Apply the filter to the signal
    filtered_sig = lfilter(b, a, sig)

    return filtered_sig


class LoRaPHY:
    def __init__(self, rf_freq, sf, bw, fs):
        if np.__version__ < '1.20.0':
            raise ValueError("Error. Newer version of NumPy is required ( >=1.20.0 ).")

        self.rf_freq = rf_freq
        self.sf = sf
        self.bw = bw
        self.fs = fs
        self.has_header = 1
        self.crc = 1
        self.fast_mode = False
        self.is_debug = False
        self.hamming_decoding_en = True
        self.zero_padding_ratio = 10
        self.cfo = 0

        self.whitening_seq = np.array([0xff, 0xfe, 0xfc, 0xf8, 0xf0, 0xe1, 0xc2, 0x85, 0xb, 0x17, 0x2f, 0x5e, 0xbc, 0x78, 0xf1, 0xe3, 0xc6, 0x8d, 0x1a, 0x34, 0x68, 0xd0, 0xa0, 0x40, 0x80, 0x1, 0x2, 0x4, 0x8, 0x11, 0x23, 0x47, 0x8e, 0x1c, 0x38, 0x71, 0xe2, 0xc4, 0x89, 0x12, 0x25, 0x4b, 0x97, 0x2e, 0x5c, 0xb8, 0x70, 0xe0, 0xc0, 0x81, 0x3, 0x6, 0xc, 0x19, 0x32, 0x64, 0xc9, 0x92, 0x24, 0x49, 0x93, 0x26, 0x4d, 0x9b, 0x37, 0x6e, 0xdc, 0xb9, 0x72, 0xe4, 0xc8, 0x90, 0x20, 0x41, 0x82, 0x5, 0xa, 0x15, 0x2b, 0x56, 0xad, 0x5b, 0xb6, 0x6d, 0xda, 0xb5, 0x6b, 0xd6, 0xac, 0x59, 0xb2, 0x65, 0xcb, 0x96, 0x2c, 0x58, 0xb0, 0x61, 0xc3, 0x87, 0xf, 0x1f, 0x3e, 0x7d, 0xfb, 0xf6, 0xed, 0xdb, 0xb7, 0x6f, 0xde, 0xbd, 0x7a, 0xf5, 0xeb, 0xd7, 0xae, 0x5d, 0xba, 0x74, 0xe8, 0xd1, 0xa2, 0x44, 0x88, 0x10, 0x21, 0x43, 0x86, 0xd, 0x1b, 0x36, 0x6c, 0xd8, 0xb1, 0x63, 0xc7, 0x8f, 0x1e, 0x3c, 0x79, 0xf3, 0xe7, 0xce, 0x9c, 0x39, 0x73, 0xe6, 0xcc, 0x98, 0x31, 0x62, 0xc5, 0x8b, 0x16, 0x2d, 0x5a, 0xb4, 0x69, 0xd2, 0xa4, 0x48, 0x91, 0x22, 0x45, 0x8a, 0x14, 0x29, 0x52, 0xa5, 0x4a, 0x95, 0x2a, 0x54, 0xa9, 0x53, 0xa7, 0x4e, 0x9d, 0x3b, 0x77, 0xee, 0xdd, 0xbb, 0x76, 0xec, 0xd9, 0xb3, 0x67, 0xcf, 0x9e, 0x3d, 0x7b, 0xf7, 0xef, 0xdf, 0xbf, 0x7e, 0xfd, 0xfa, 0xf4, 0xe9, 0xd3, 0xa6, 0x4c, 0x99, 0x33, 0x66, 0xcd, 0x9a, 0x35, 0x6a, 0xd4, 0xa8, 0x51, 0xa3, 0x46, 0x8c, 0x18, 0x30, 0x60, 0xc1, 0x83, 0x7, 0xe, 0x1d, 0x3a, 0x75, 0xea, 0xd5, 0xaa, 0x55, 0xab, 0x57, 0xaf, 0x5f, 0xbe, 0x7c, 0xf9, 0xf2, 0xe5, 0xca, 0x94, 0x28, 0x50, 0xa1, 0x42, 0x84, 0x9, 0x13, 0x27, 0x4f, 0x9f, 0x3f, 0x7f], dtype=np.uint8)

        self.header_checksum_matrix = np.array([
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1]
        ], dtype=np.uint8)

        # CRC generator initialization (Assuming Python equivalent exists)
        #self.crc_generator = None  # You would need to replace this with an actual CRC generator in Python.
        #self.crc_generator = crcmod.mkCrcFun(int("x16 + x12 + x5 + 1", 2))
        
        self.preamble_len = 6

        self.init()
        
    def crc_generator(self, data: bytes):
        '''
        CRC-16-ModBus Algorithm
        '''
        data = bytearray(data)
        poly = 0xA001
        crc = 0xFFFF
        for b in data:
            crc ^= (0xFF & b)
            for _ in range(0, 8):
                if (crc & 0x0001):
                    crc = ((crc >> 1) & 0xFFFF) ^ poly
                else:
                    crc = ((crc >> 1) & 0xFFFF)
        return np.uint16(crc)
        #return crc
    """
    def crc_generator(self, data):
        polynomial='x16+x12+x5+1'
        crc_func = crcmod.predefined.mkCrcFun(polynomial)
        crc_value = crc_func(data.encode('utf-8'))
        return format(crc_value, '04X')
        #crcmod.mkCrcFun(int("x16 + x12 + x5 + 1", 2))
    """
    def init(self):
        self.bin_num = 2**self.sf * self.zero_padding_ratio
        self.sample_num = 2 * 2**self.sf
        self.fft_len = self.sample_num * self.zero_padding_ratio

        # Chirp generation (You would need to implement the chirp generation in Python)

        if 2**(self.sf) / self.bw > 16e-3:
            self.ldr = 1
        else:
            self.ldr = 0

    """
    @staticmethod
    def chirp(is_upchirp, sf, bw, total_bw, t, cfo, snr):
        # Implement chirp generation here
        pass
    """

    def dechirp(self, x, is_up=False):
        if not is_up:
            c = self.upchirp
        else:
            c = self.downchirp

        ft = np.fft.fft(self.sig[x:x+self.sample_num] * c, self.fft_len)
        ft_ = np.abs(ft[0:self.bin_num]) + np.abs(ft[self.fft_len-self.bin_num:self.fft_len])
        pk = self.topn(np.column_stack((ft_, np.arange(1, self.bin_num + 1))), 1)
        return pk

    def detect(self, start_idx):
        ii = start_idx
        pk_bin_list = []  # preamble peak bin list
        while ii < len(self.sig) - self.sample_num * self.preamble_len:
            # search preamble_len-1 basic upchirps
            if len(pk_bin_list) == self.preamble_len - 1:
                x = ii - round((pk_bin_list[-1] - 1) / (self.zero_padding_ratio * 2))
                return x

            pk0 = self.dechirp(ii)
            if pk_bin_list:
                bin_diff = (pk_bin_list[-1] - pk0[1]) % self.bin_num
                if bin_diff > self.bin_num / 2:
                    bin_diff = self.bin_num - bin_diff
                if bin_diff <= self.zero_padding_ratio:
                    pk_bin_list.append(pk0[1])
                else:
                    pk_bin_list = [pk0[1]]
            else:
                pk_bin_list = [pk0[1]]

            ii = ii + self.sample_num

        return -1

    def demodulate(self, sig):
        self.cfo = 0
        self.init()

        if not self.fast_mode:
            sig = lfilter([1], [1, -1], sig, axis=0)
            sig = resample(sig, int(2 * self.bw), self.fs, axis=0)

        self.sig = sig
        symbols_m = []
        cfo_m = []
        netid_m = []

        x = 0
        while x < len(self.sig):
            x = self.detect(x)
            if x < 0:
                break

            # align symbols with SFD
            x = self.sync(x)

            # NetID
            pk_netid1 = self.dechirp(int(x - 4.25 * self.sample_num))
            pk_netid2 = self.dechirp(int(x - 3.25 * self.sample_num))
            netid_m.append([(pk_netid1[1] + self.bin_num - self.preamble_bin) // self.zero_padding_ratio % (2 ** self.sf),
                             (pk_netid2[1] + self.bin_num - self.preamble_bin) // self.zero_padding_ratio % (2 ** self.sf)])

            # the goal is to extract payload_len from PHY header
            # header is in the first 8 symbols
            symbols = []
            pk_list = []
            if x > len(self.sig) - 8 * self.sample_num + 1:
                return symbols_m, cfo_m, netid_m

            for ii in range(8):
                pk = self.dechirp(x + ii * self.sample_num)
                pk_list.append(pk)
                symbols.append((pk[1] + self.bin_num - self.preamble_bin) // self.zero_padding_ratio % (2 ** self.sf))

            if self.has_header:
                is_valid = self.parse_header(symbols)
                if not is_valid:
                    x = x + 7 * self.sample_num
                    continue

            # number of symbols in the packet
            sym_num = self.calc_sym_num(self.payload_len)

            # demodulate the rest LoRa data symbols
            if x > len(self.sig) - sym_num * self.sample_num + 1:
                return symbols_m, cfo_m, netid_m

            for ii in range(8, sym_num):
                pk = self.dechirp(x + ii * self.sample_num)
                pk_list.append(pk)
                symbols.append((pk[1] + self.bin_num - self.preamble_bin) // self.zero_padding_ratio % (2 ** self.sf))
            x = x + sym_num * self.sample_num

            # compensate CFO drift
            symbols = self.dynamic_compensation(symbols)

            symbols_m.extend([int(round(symb)) % (2 ** self.sf) for symb in symbols])
            cfo_m.extend([self.cfo] * len(symbols))

        if not symbols_m:
            print('No preamble detected!')
        return symbols_m, cfo_m, netid_m
    """
    def parse_header(self, data):
        symbols = self.dynamic_compensation(data)
        symbols_g = self.gray_coding(symbols)

        codewords = self.diag_deinterleave(symbols_g[:8], self.sf - 2)

        nibbles = self.hamming_decode(codewords, 8)
        self.payload_len = int(nibbles[0] * 16 + nibbles[1])
        self.crc = int(nibbles[2] & 1)
        self.cr = int(nibbles[2] >> 1)

        header_checksum = [int(nibbles[3] & 1)] + list(np.flip(np.binary_repr(nibbles[4], width=4), axis=0))
        header_checksum_calc = self.header_checksum_matrix.dot(gf(np.reshape(np.binary_repr(nibbles[:3], width=4), (-1, 1))))
        
        if any(header_checksum != header_checksum_calc):
            print('Invalid header checksum!')
            is_valid = 0
        else:
            is_valid = 1

        return is_valid
    """
    def parse_header(self, data):
        # Compensate CFO drift
        symbols = self.dynamic_compensation(data)

        # Gray coding
        symbols_g = self.gray_coding(symbols)

        # Deinterleave
        codewords = self.diag_deinterleave(symbols_g[0:8], self.sf - 2)
        
        # Parse header
        nibbles = self.hamming_decode(codewords, 8)
        self.payload_len = int(nibbles[0] * 16 + nibbles[1])
        self.crc = int(nibbles[2] & 1)
        self.cr = int(nibbles[2] >> 1)
        
        # Calculate header checksum on the first three nibbles
        # The valid header checksum is considered to be 5 bits
        # Other 3 bits require further reverse engineering
        header_checksum = [nibbles[3] & 1] + list(np.unpackbits(np.uint8(nibbles[4])))[4:]
        header_checksum_calc = np.dot(self.header_checksum_matrix, np.reshape(np.unpackbits(np.uint8(nibbles[0:3])), [-1, 1]))

        if not np.array_equal(header_checksum, header_checksum_calc):
            print('Invalid header checksum!')
            is_valid = False
        else:
            is_valid = True

        return is_valid

    def modulate(self, symbols):
        uc = self.chirp(True, self.sf, self.bw, self.fs, 0, self.cfo, 0)
        dc = self.chirp(False, self.sf, self.bw, self.fs, 0, self.cfo, 0)
        preamble = np.tile(uc, (self.preamble_len, 1))
        netid = np.concatenate((self.chirp(True, self.sf, self.bw, self.fs, 24, self.cfo, 0),
                                self.chirp(True, self.sf, self.bw, self.fs, 32, self.cfo, 0)))

        chirp_len = len(uc)
        sfd = np.concatenate((dc, dc, dc[:round(chirp_len/4)]))
        data = np.zeros(len(symbols) * chirp_len, dtype=np.float32)
        
        for i, symbol in enumerate(symbols):
            data[i * chirp_len:(i + 1) * chirp_len] = self.chirp(True, self.sf, self.bw, self.fs, symbol, self.cfo, 0)

        s = np.concatenate((preamble, netid, sfd, data))
        return s

    def encode(self, payload):
        if self.crc:
            data = np.concatenate((payload, self.calc_crc(payload)))
        else:
            data = payload

        plen = len(payload)
        sym_num = self.calc_sym_num(plen)
        nibble_num = self.sf - 2 + (sym_num - 8) / (self.cr + 4) * (self.sf - 2 * self.ldr)
        data_w = np.concatenate((data, 255 * np.ones(int(np.ceil((nibble_num - 2 * len(data)) / 2)))))
        data_w[:plen] = self.whiten(data_w[:plen])
        data_nibbles = np.zeros(int(nibble_num), dtype=np.uint8)

        for i in range(int(nibble_num)):
            idx = int(np.ceil(i / 2))
            if i % 2 == 0:
                data_nibbles[i] = data_w[idx] & 0xF
            else:
                data_nibbles[i] = data_w[idx] >> 4

        if self.has_header:
            header_nibbles = self.gen_header(plen)
        else:
            header_nibbles = []

        codewords = self.hamming_encode(np.concatenate((header_nibbles, data_nibbles)))

        symbols_i = self.diag_interleave(codewords[:self.sf - 2], 8)
        ppm = self.sf - 2 * self.ldr
        rdd = self.cr + 4
        
        for i in range(self.sf - 1, len(codewords) - ppm + 1, ppm):
            symbols_i = np.concatenate((symbols_i, self.diag_interleave(codewords[i:i + ppm], rdd)))

        symbols = self.gray_decoding(symbols_i)
        return symbols
    """
    def gen_header(self, plen):
        header_nibbles = np.zeros(5, dtype=np.uint8)
        header_nibbles[0] = plen >> 4
        header_nibbles[1] = plen & 0xF
        header_nibbles[2] = (2 * self.cr) | self.crc

        header_checksum = self.header_checksum_matrix.dot(gf(np.reshape(np.binary_repr(header_nibbles[:3], width=4), (-1, 1))))
        x = header_checksum.x
        header_nibbles[3] = x[0]

        for i in range(4):
            header_nibbles[4] |= x[i + 1] << (3 - i)

        return header_nibbles
    """
    def gen_header(self, plen):
        # Initialize header nibbles
        header_nibbles = np.zeros(5, dtype=int)

        # Calculate the first two header nibbles
        header_nibbles[0] = plen >> 4
        header_nibbles[1] = plen & 0b1111

        # Calculate the third header nibble
        header_nibbles[2] = (2 * self.cr) | self.crc

        # Calculate header checksum
        header_checksum = np.dot(self.header_checksum_matrix, np.reshape(np.unpackbits(np.uint8(header_nibbles[0:3])), [-1, 1]))
        x = header_checksum.x
        header_nibbles[3] = x[0]

        # Calculate the last header nibble
        for i in range(1, 5):
            header_nibbles[4] |= (x[i] << (4 - i))

        return header_nibbles
    
    def calc_crc(self, data):
        if len(data) == 0:
            checksum = [0, 0]
        elif len(data) == 1:
            checksum = [data[-1], 0]
        elif len(data) == 2:
            checksum = [data[-1], data[-2]]
        else:
            input_data = data[:-2]
            input_bytes = bytearray(input_data)

            #print(len(input_data))
            #print(input_data)
            crc = self.crc_generator(input_bytes)
            #crc = crc16.xmodem(input_bytes)
            #print(crc)
            #print(type(crc))
            checksum_b1 = crc & 0xFF
            checksum_b2 = (crc >> 8) & 0xFF
            checksum = [checksum_b1, checksum_b2]
            
        return checksum

    def whiten(self, data):
        data_len = len(data)
        print(data)
        data_w = data[:data_len] ^ self.whitening_seq[:data_len]
        self.print_bin("Whiten", data_w)
        return data_w

    def hamming_encode(self, nibbles):
        nibble_num = len(nibbles)
        codewords = np.zeros(nibble_num, dtype=np.uint8)

        for i in range(nibble_num):
            nibble = nibbles[i]
            p1 = self.bit_reduce(np.bitwise_xor, nibble, [1, 3, 4])
            p2 = self.bit_reduce(np.bitwise_xor, nibble, [1, 2, 4])
            p3 = self.bit_reduce(np.bitwise_xor, nibble, [1, 2, 3])
            p4 = self.bit_reduce(np.bitwise_xor, nibble, [1, 2, 3, 4])
            p5 = self.bit_reduce(np.bitwise_xor, nibble, [2, 3, 4])

            if i < self.sf - 2:
                cr_now = 4
            else:
                cr_now = self.cr

            if cr_now == 1:
                codewords[i] = (p4 << 4) | nibble
            elif cr_now == 2:
                codewords[i] = self.word_reduce(np.bitwise_or, [(p5 << 5), (p3 << 4), nibble])
            elif cr_now == 3:
                codewords[i] = self.word_reduce(np.bitwise_or, [(p2 << 6), (p5 << 5), (p3 << 4), nibble])
            elif cr_now == 4:
                codewords[i] = self.word_reduce(np.bitwise_or, [(p1 << 7), (p2 << 6), (p5 << 5), (p3 << 4), nibble])
            else:
                raise ValueError('Invalid Code Rate!')

        return codewords

    def diag_interleave(self, codewords, rdd):
        tmp = np.unpackbits(codewords.reshape(-1, 1), axis=1)[:, -rdd:]
        symbols_i = np.packbits(np.array([np.roll(tmp[:, i], 1 - i) for i in range(rdd)], dtype=np.uint8).T)
        self.print_bin("Interleave", symbols_i)
        return symbols_i

    def gray_decoding(self, symbols_i):
        symbols = np.zeros(len(symbols_i), dtype=np.uint8)

        for i in range(len(symbols_i)):
            num = symbols_i[i].astype(np.uint16)
            mask = num >> 1
            while mask != 0:
                num = num ^ mask
                mask = mask >> 1

            if i < 8 or self.ldr:
                symbols[i] = (num * 4 + 1) % (2 ** self.sf)
            else:
                symbols[i] = (num + 1) % (2 ** self.sf)

        return symbols
    
    def calc_sym_num(self, plen):
        sym_num = 8 + max((4 + self.cr) * np.ceil((2 * plen - self.sf + 7 + 4 * self.crc - 5 * (1 - self.has_header)) / (self.sf - 2 * self.ldr)), 0)
        return int(sym_num)

    def calc_payload_len(self, slen, no_redundant_bytes=False):
        if not no_redundant_bytes:
            plen_float = (self.sf - 2) / 2 - 2.5 * self.has_header + (self.sf - self.ldr * 2) / 2 * np.ceil((slen - 8) / (self.cr + 4))
            plen = int(np.floor(plen_float))
        else:
            plen_float = (self.sf - 2) / 2 - 2.5 * self.has_header + (self.sf - self.ldr * 2) / 2 * np.ceil((slen - 8) / (self.cr + 4))
            plen = int(np.ceil(plen_float))
        return plen

    def sync(self, x):
        found = False
        while x < len(self.sig) - self.sample_num:
            up_peak = self.dechirp(x)
            down_peak = self.dechirp(x, False)
            if abs(down_peak[0]) > abs(up_peak[0]):
                found = True
            x = x + self.sample_num
            if found:
                break

        if not found:
            return None

        pkd = self.dechirp(x, False)
        if pkd[1] > self.bin_num / 2:
            to = round((pkd[1] - 1 - self.bin_num) / self.zero_padding_ratio)
        else:
            to = round((pkd[1] - 1) / self.zero_padding_ratio)
        x = x + to

        pku = self.dechirp(x - 4 * self.sample_num)
        self.preamble_bin = pku[1]

        if self.preamble_bin > self.bin_num / 2:
            self.cfo = (self.preamble_bin - self.bin_num - 1) * self.bw / self.bin_num
        else:
            self.cfo = (self.preamble_bin - 1) * self.bw / self.bin_num

        pku = self.dechirp(x - self.sample_num)
        pkd = self.dechirp(x - self.sample_num, False)
        if abs(pku[0]) > abs(pkd[0]):
            x_sync = x + round(2.25 * self.sample_num)
        else:
            x_sync = x + round(1.25 * self.sample_num)
        return x_sync
    """
    def decode(self, symbols_m):
        data_m = []
        checksum_m = []

        for pkt_num in range(symbols_m.shape[1]):
            symbols_g = self.gray_coding(symbols_m[:, pkt_num])

            codewords = self.diag_deinterleave(symbols_g[0:8], self.sf - 2)
            if not self.has_header:
                nibbles = self.hamming_decode(codewords, 8)
            else:
                nibbles = self.hamming_decode(codewords, 8)
                self.payload_len = int(nibbles[0] * 16 + nibbles[1])
                self.crc = int(nibbles[2] & 1)
                self.cr = int(nibbles[2] >> 1)
                header_checksum = [int(nibbles[3] & 1)] + list(np.binary_repr(nibbles[4], width=4))
                header_checksum_calc = self.header_checksum_matrix.dot(gf(np.reshape(np.binary_repr(nibbles[0:3], width=4), (-1, 1))))
                if any(header_checksum != header_checksum_calc):
                    raise ValueError('Invalid header checksum!')
                nibbles = nibbles[5:]

            rdd = self.cr + 4
            for ii in range(9, len(symbols_g), rdd):
                codewords = self.diag_deinterleave(symbols_g[ii:ii + rdd], self.sf - 2 * self.ldr)
                nibbles = np.concatenate((nibbles, self.hamming_decode(codewords, rdd)))

            bytes_ = np.zeros(min(255, int(np.floor(len(nibbles) / 2))), dtype=np.uint8)
            for ii in range(len(bytes_)):
                bytes_[ii] = int(nibbles[2 * ii]) | (16 * int(nibbles[2 * ii + 1]))

            len_ = self.payload_len
            if self.crc:
                data = np.concatenate((self.dewhiten(bytes_[0:len_]), bytes_[len_:len_ + 2]))
                checksum = self.calc_crc(data[0:len_])
            else:
                data = self.dewhiten(bytes_[0:len_])
                checksum = []

            data_m.append(data.tolist())
            checksum_m.append(checksum.tolist())

        return data_m, checksum_m
    """
    def decode(self, symbols_m):
        data_m = []
        checksum_m = []

        for pkt_num in range(symbols_m.shape[1]):
            # Gray coding
            symbols_g = self.gray_coding(symbols_m[:, pkt_num])

            # Deinterleave
            codewords = self.diag_deinterleave(symbols_g[0:8], self.sf - 2)
            if not self.has_header:
                nibbles = self.hamming_decode(codewords, 8)
            else:
                # Parse header
                nibbles = self.hamming_decode(codewords, 8)
                self.payload_len = int(nibbles[0] * 16 + nibbles[1])
                self.crc = int(nibbles[2] & 1)
                self.cr = int(nibbles[2] >> 1)
                # Calculate header checksum on the first three nibbles
                # The valid header checksum is considered to be 5 bits
                # Other 3 bits require further reverse engineering
                header_checksum = [nibbles[3] & 1] + list(np.unpackbits(np.uint8(nibbles[4])))[4:]
                header_checksum_calc = np.dot(self.header_checksum_matrix, np.reshape(np.unpackbits(np.uint8(nibbles[0:3])), [-1, 1]))
                if not np.array_equal(header_checksum, header_checksum_calc):
                    raise ValueError('Invalid header checksum!')
                nibbles = nibbles[5:]

            rdd = self.cr + 4
            for ii in range(9, len(symbols_g) - rdd + 1, rdd):
                codewords = self.diag_deinterleave(symbols_g[ii:ii+rdd], self.sf - 2 * self.ldr)
                # Hamming decode
                nibbles = np.concatenate((nibbles, self.hamming_decode(codewords, rdd)))

            # Combine nibbles to bytes
            bytes = np.zeros(min(255, int(np.floor(len(nibbles) / 2))), dtype=np.uint8)
            for ii in range(len(bytes)):
                bytes[ii] = np.uint8(nibbles[2 * ii] | (16 * nibbles[2 * ii + 1]))

            # Dewhitening
            payload_len = self.payload_len
            if self.crc:
                # The last 2 bytes are CRC16 checksum
                data = np.concatenate((self.dewhiten(bytes[0:payload_len]), bytes[payload_len:payload_len+2]))
                # Calculate CRC checksum
                checksum = self.calc_crc(data[0:payload_len])
            else:
                data = self.dewhiten(bytes[0:payload_len])
                checksum = []

            data_m.extend(data.tolist())
            checksum_m.extend(checksum.tolist())

        return data_m, checksum_m
    def dynamic_compensation(self, data):
        sfo_drift = (1 + np.arange(len(data))) * (2 ** self.sf) * self.cfo / self.rf_freq
        symbols = np.mod(data - sfo_drift, 2 ** self.sf)

        if self.ldr:
            bin_offset = 0
            v_last = 1

            for i in range(len(symbols)):
                v = symbols[i]
                bin_delta = np.mod(v - v_last, 4)
                if bin_delta < 2:
                    bin_offset = bin_offset - bin_delta
                else:
                    bin_offset = bin_offset - bin_delta + 4
                v_last = v
                symbols[i] = np.mod(v + bin_offset, 2 ** self.sf)

        return symbols

    def gray_coding(self, din):
        din[0:8] = np.floor(din[0:8] / 4)
        if self.ldr:
            din[8:] = np.floor(din[8:] / 4)
        else:
            din[8:] = np.mod(din[8:] - 1, 2 ** self.sf)
        s = din.astype(np.uint16)
        symbols = np.bitwise_xor(s, np.right_shift(s, 1))
        self.print_bin("Gray Coding", symbols, self.sf)
        return symbols
    
    def diag_deinterleave(self, symbols, ppm):
        b = np.unpackbits(symbols.astype(np.uint8))
        b = np.flipud(b.reshape(-1, ppm))
        codewords = np.packbits(b)
        self.print_bin("Deinterleave", codewords)
        return codewords

    def dewhiten(self, bytes):
        len_ = len(bytes)
        bytes_w = np.bitwise_xor(bytes.astype(np.uint8), self.whitening_seq[0:len_])
        self.print_bin("Dewhiten", bytes_w)
        return bytes_w

    def hamming_decode(self, codewords, rdd):
        p1 = self.bit_reduce(np.bitwise_xor, codewords, [8, 4, 3, 1])
        p2 = self.bit_reduce(np.bitwise_xor, codewords, [7, 4, 2, 1])
        p3 = self.bit_reduce(np.bitwise_xor, codewords, [5, 3, 2, 1])
        p4 = self.bit_reduce(np.bitwise_xor, codewords, [5, 4, 3, 2, 1])
        p5 = self.bit_reduce(np.bitwise_xor, codewords, [6, 4, 3, 2])

        def parity_fix(p):
            if p == 3:
                return 4
            elif p == 5:
                return 8
            elif p == 6:
                return 1
            elif p == 7:
                return 2
            else:
                return 0

        if self.hamming_decoding_en:
            if rdd in [5, 6]:
                nibbles = np.mod(codewords, 16)
            elif rdd in [7, 8]:
                parity = p2 * 4 + p3 * 2 + p5
                pf = np.array([parity_fix(p) for p in parity])
                codewords = np.bitwise_xor(codewords, pf.astype(np.uint16))
                nibbles = np.mod(codewords, 16)
            else:
                raise ValueError('Invalid Code Rate!')
        else:
            nibbles = np.mod(codewords, 16)
        self.print_bin("Hamming Decode", codewords)
        return nibbles

    def symbols_to_bytes(self, symbols):
        self.init()
        self.hamming_decoding_en = False
        payload_len_ = self.payload_len

        if len(symbols) <= 4:
            slen_tmp = 8 + self.has_header * (self.cr + 4)
        else:
            slen_tmp = 8 + np.ceil((len(symbols) - 4 * (1 - self.has_header)) / 4) * (self.cr + 4)
        self.payload_len = self.calc_payload_len(slen_tmp, True)
        symbols_ = np.zeros(self.calc_sym_num(self.payload_len))

        if self.has_header:
            jj = 9
        else:
            jj = 1

        for ii in range(0, len(symbols), 4):
            if ii + 3 <= len(symbols):
                symbols_[jj:jj + 4] = symbols[ii:ii + 4]
            else:
                symbols_[jj:jj + 4] = np.concatenate((symbols[ii:], np.zeros(ii - len(symbols) + 3)))
            if jj == 1:
                jj = 9
            else:
                jj = jj + self.cr + 4

        if self.has_header:
            symbols_tmp = self.encode(np.zeros(self.payload_len))
            symbols_[0:8] = symbols_tmp[0:8]

        bytes_, _ = self.decode(symbols_)

        if self.crc:
            bytes_ = bytes_[:-2]

        self.hamming_decoding_en = True
        self.payload_len = payload_len_
        return bytes_

    def time_on_air(self, plen):
        sym_num = self.calc_sym_num(plen)
        time_ms = (sym_num + 4.25 + self.preamble_len) * (2 ** self.sf / self.bw) * 1000
        return time_ms
    
    def print_bin(self, flag, vec, size=8):
        if self.is_debug:
            print(f"{flag}:")
            for value in vec:
                print(format(int(round(value)), f'0{size}b'))
            print()

    def print_hex(self, flag, vec):
        if self.is_debug:
            print(f"{flag}: ", end="")
            for value in vec:
                print(format(int(round(value)), '02X'), end=" ")
            print()

    def log(self, flag, data):
        if self.is_debug:
            print(f"{flag}: ", end="")
            for value in data:
                print(value, end=" ")
            print()

    def plot_peak(self, x):
        

        fig, axs = plt.subplots(2, 9, figsize=(18, 6))

        c = np.concatenate((self.downchirp, self.upchirp), axis=1)

        for jj in range(9):
            for ii in range(2):
                ft = np.fft.fft(self.sig[x:x + self.sample_num] * c[:, ii], self.fft_len)
                ft_ = np.abs(ft[0:self.bin_num]) + np.abs(ft[self.fft_len - self.bin_num:self.fft_len])
                axs[ii, jj].plot(ft_)
            x += self.sample_num

        plt.show()
        
    def bit_reduce(fn, w, pos):
        b = bool((w & (1 << pos[0])) >> pos[0])
        for i in range(1, len(pos)):
            b = fn(b, bool((w & (1 << pos[i])) >> pos[i]))
        return b

    @staticmethod
    def word_reduce(fn, ws):
        w = ws[0]
        for i in range(1, len(ws)):
            w = fn(w, ws[i])
        return w

    @staticmethod
    def topn(pks, n, padding=True, th=None):
        p = abs(pks[:, 0])
        idx = (-p).argsort()
        nn = min(n, pks.shape[0])
        if padding:
            y = np.vstack((pks[idx[:nn], :], np.zeros((n - nn, pks.shape[1]))))
        else:
            y = pks[idx[:nn], :]

        if th is not None:
            idx = np.where(np.abs(y[:, 0]) >= th)[0]
            y = y[:idx[0], :] if idx.size > 0 else np.zeros((0, pks.shape[1]))

        return y

    @staticmethod
    def chirp(is_up, sf, bw, fs, h, cfo=0, tdelta=0, tscale=1):
        if tscale is None:
            tscale = 1
        if tdelta is None:
            tdelta = 0
        if cfo is None:
            cfo = 0

        N = 2 ** sf
        T = N / bw
        samp_per_sym = round(fs / bw * N)
        h_orig = h
        h = round(h)
        cfo = cfo + (h_orig - h) / N * bw

        if is_up:
            k = bw / T
            f0 = -bw / 2 + cfo
        else:
            k = -bw / T
            f0 = bw / 2 + cfo

        t = (np.arange(samp_per_sym * (N - h) // N) / fs * tscale + tdelta).reshape(-1, 1)
        snum = len(t)
        c1 = np.exp(1j * 2 * np.pi * (t * (f0 + k * T * h / N + 0.5 * k * t)))

        phi = 0 if snum == 0 else np.angle(c1[snum - 1])
        t = (np.arange(samp_per_sym * h // N) / fs + tdelta).reshape(-1, 1)
        c2 = np.exp(1j * (phi + 2 * np.pi * (t * (f0 + 0.5 * k * t))))

        y = np.concatenate((c1[:snum - 1], c2), axis=1).T
        return y
    
    @staticmethod
    def spec(sig, fs, bw, sf):
        import numpy as np
        import matplotlib.pyplot as plt

        x = np.arange(0, 2 ** sf / bw + 1 / sf, 1 / sf)
        y = np.arange(-bw / 2, bw / 2, bw / (2 ** sf))
        p = fs / bw
        win_length = 2 ** (sf - 2)
        N = p * 2 ** sf

        s = np.abs(np.fft.fft(sig, n=N))
        valid_data_len = round(2 ** sf / 2 * 1.5)
        b = np.abs(s[valid_data_len::-1])
        c = np.abs(s[-1:-valid_data_len - 1:-1])
        d = np.vstack((b, c))
        plt.figure()
        plt.imshow(d, extent=(x.min(), x.max(), y.min(), y.max()), cmap='summer', aspect='auto')
        plt.title('Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.show()
        
    def read(filename, count=None):
        try:
            with open(filename, 'rb') as file:
                if count is None:
                    data = np.fromfile(file, dtype=np.complex64)
                else:
                    data = np.fromfile(file, dtype=np.complex64, count=count)
            return data
        except FileNotFoundError:
            print(f"File '{filename}' not found.")
            return None

    @staticmethod
    def write(data, filename):
        try:
            with open(filename, 'wb') as file:
                data.tofile(file)
        except Exception as e:
            print(f"Error writing to file '{filename}': {e}")

