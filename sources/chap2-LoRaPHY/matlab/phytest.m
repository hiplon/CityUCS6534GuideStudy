% test.m

rf_freq = 866e6;    % carrier frequency 866 MHz, used to correct clock drift
sf = 7;             % spreading factor SF7
bw = 125e3;         % bandwidth 125 kHz
fs = 1e6;           % sampling rate 1 MHz

phy = LoRaPHY(rf_freq, sf, bw, fs);
phy.has_header = 1;         % explicit header mode
phy.cr = 4;                 % code rate = 4/8 (1:4/5 2:4/6 3:4/7 4:4/8)
phy.crc = 1;                % enable payload CRC checksum
phy.preamble_len = 8;       % preamble: 8 basic upchirps

% Encode payload [1 2 3 4 5]
% symbols = phy.encode((1:5)');
% Encode payload [78 22 43 44 12]
payload = [78 22 43 44 12];
symbols = phy.encode(payload');

fprintf("[encode] symbols:\n");
%disp(symbols);

% Baseband Modulation
sig = phy.modulate(symbols);
% draw the frequency figure
LoRaPHY.spec(sig, fs, bw, sf);
% Demodulation
[symbols_d, cfo, netid] = phy.demodulate(sig);
fprintf("[demodulate] symbols:\n");
%disp(symbols_d);

% Decoding
[data, checksum] = phy.decode(symbols_d);
fprintf("[decode] data:\n");
disp(data);
fprintf("[decode] checksum:\n");
disp(checksum);