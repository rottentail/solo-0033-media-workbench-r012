"""Partitioned causal matrix convolution on the programme excitation clock.

The measured room is a 2x2 stereo matrix whose four impulse responses are kept
sample-by-sample with their signed amplitudes and interchannel calibration.
Uniform-partition overlap-save convolution runs on a fixed internal partition
grid, so output block sizes never change reflection timing, and every filter
keeps its own state across the whole programme.  Room transitions fade the
*excitation* entering each room while the already-excited reflections continue
to decay naturally inside that room's convolver.
"""
from array import array
from . import audio_signal, room_plan
from .denoise_signal import fft

# Internal overlap-save partition.  The measured responses are at most one
# second (48000 taps); six partitions cover that length with bounded spectra.
PARTITION = 8192


def pack(spectrum):
    """Store an FFT spectrum as an interleaved float32 pair array."""
    packed = array('f', [0.0])*(len(spectrum)*2)
    for index, value in enumerate(spectrum):
        packed[2*index] = value.real
        packed[2*index+1] = value.imag
    return packed


def weights(stages, sample):
    """Stage excitation coefficients at one programme sample."""
    result = [0.0]*len(stages)
    active = 0
    while active+1 < len(stages) and sample >= stages[active+1]['at']:
        active += 1
    stage = stages[active]
    if active and stage['fade'] and sample < stage['at']+stage['fade']:
        fraction = (sample-stage['at'])/stage['fade']
        result[active-1], result[active] = 1-fraction, fraction
    else:
        result[active] = 1.0
    return result


def _coeff_table(stages, origin, count):
    """Per-sample excitation coefficients, evaluated on the sample clock."""
    table = [[0.0]*count for _ in stages]
    active = 0
    for local in range(count):
        sample = origin+local
        while active+1 < len(stages) and sample >= stages[active+1]['at']:
            active += 1
        stage = stages[active]
        if active and stage['fade'] and sample < stage['at']+stage['fade']:
            fraction = (sample-stage['at'])/stage['fade']
            table[active-1][local] = 1-fraction
            table[active][local] = fraction
        else:
            table[active][local] = 1.0
    return table


class Matrix:
    """One 2x2 measured room convolved with uniform overlap-save partitions.

    Uniform-partition overlap-save: each kernel partition is zero-padded and
    FFT'd from index zero; input block tau is windowed as ``[X(tau-1), X(tau)]``
    (previous partition first, current partition second).  The alias-free
    causal output block is the second half of the IFFT of
    ``sum_k H(k) * FFT(X(tau-k))``; the newest input spectrum is history index
    zero.
    """

    def __init__(self, kernels, partition=PARTITION):
        if type(partition) is not int or partition < 1 or partition & (partition-1):
            raise ValueError('room partition must be a positive power of two')
        self.partition = partition
        self.size = partition*2
        taps = room_plan.kernels(kernels) if isinstance(kernels, dict) else kernels
        if (len(taps) != 4 or any(len(channel) != len(taps[0]) for channel in taps)
                or not taps[0]):
            raise ValueError('room matrix needs four equally long impulse responses')
        self.frames = len(taps[0])
        self.kernel_parts = (self.frames+partition-1)//partition
        self._spectra = []
        for channel in range(4):
            parts = []
            for k in range(self.kernel_parts):
                # Every sub-filter starts at index zero; the partition delay
                # lives in the input-block pairing, not in the window placement.
                window = [0.0]*self.size
                segment = taps[channel][k*partition:(k+1)*partition]
                window[:len(segment)] = segment
                parts.append(pack(fft(window)))
            self._spectra.append(parts)
        # The packed float32 spectra are the only retained filter state; the
        # decoded taps are released so memory stays bounded by the partitions.
        del taps
        # Newest input spectrum first; at most one entry per kernel partition.
        self._history = [[], []]
        self._previous = [[0.0]*partition for _ in range(2)]

    def advance(self, excitation, coefficient):
        """Process one partition.  ``coefficient`` may be a scalar or a
        per-sample list of excitation weights for this room."""
        if len(excitation) != 2 or any(len(row) != self.partition for row in excitation):
            raise ValueError('room matrix processes one full stereo partition')
        if isinstance(coefficient, (int, float)):
            scaled = [[value*coefficient for value in excitation[c]] for c in range(2)]
        else:
            if len(coefficient) != self.partition:
                raise ValueError('one excitation coefficient per sample required')
            scaled = [[excitation[c][i]*coefficient[i] for i in range(self.partition)]
                      for c in range(2)]
        re_out = [[0.0]*self.size for _ in range(2)]
        im_out = [[0.0]*self.size for _ in range(2)]
        for channel in range(2):
            # Window [previous partition | current weighted partition].  The
            # previous block always shifts through, even when this room receives
            # no excitation now, so its in-flight reflections keep decaying.
            block = scaled[channel]
            window = [0.0]*self.size
            window[:self.partition] = self._previous[channel]
            window[self.partition:] = block
            self._previous[channel] = block
            history = self._history[channel]
            # Even an all-zero block goes through the full window so its first
            # half (the previous partition) keeps contributing in-flight taps.
            history.insert(0, pack(fft(window)))
            if len(history) > self.kernel_parts:
                del history[self.kernel_parts:]
            for out in range(2):
                transfers = self._spectra[out*2+channel]
                acc_re, acc_im = re_out[out], im_out[out]
                # H(k) pairs with the input block k partitions older than the
                # current one; unseen prehistory contributes nothing.
                for k in range(min(len(history), self.kernel_parts)):
                    block_spectrum = history[k]
                    transfer = transfers[k]
                    for bin_index in range(self.size):
                        base = bin_index+bin_index
                        xr = block_spectrum[base]
                        xi = block_spectrum[base+1]
                        hr = transfer[base]
                        hi = transfer[base+1]
                        acc_re[bin_index] += xr*hr-xi*hi
                        acc_im[bin_index] += xr*hi+xi*hr
        result = []
        for out in range(2):
            inverse = fft([complex(re_out[out][i], im_out[out][i]) for i in range(self.size)], True)
            result.append([value.real for value in inverse][self.partition:])
        return result

    def flush(self):
        """Drain this room with one partition of zero excitation."""
        return self.advance([[0.0]*self.partition for _ in range(2)], 0.0)

    @property
    def partitions(self):
        # Zero blocks requested after the last real input partition so every
        # kernel partition finishes contributing its measured reflections.
        return self.kernel_parts


def _read_partition(source):
    """Read one partition, padded with trailing zeros at programme end."""
    raw = audio_signal.read_frames(source, PARTITION)
    if len(raw) > PARTITION*2:
        raise ValueError('programme spool overread')
    frames = len(raw)//2
    channels = [[0.0]*PARTITION for _ in range(2)]
    for index in range(frames):
        channels[0][index] = raw[index*2]
        channels[1][index] = raw[index*2+1]
    return channels, frames


def blocks(source, plan, block_frames=4096):
    """Render the complete programme and its full measured decay.

    The convolver always starts at programme sample zero with empty state, so a
    mid-programme slice rendered from the same frozen plan is bit-identical to
    that position in the full delivery.  ``block_frames`` only regroups the
    emitted samples; the internal partition grid is fixed.
    """
    if type(block_frames) is not int or not 1 <= block_frames <= 65536:
        raise ValueError('invalid room output block size')
    rooms = [Matrix(stage['response']) for stage in plan['stages']]
    dry, wet_gain = plan['dry'], plan['wet']
    total = plan['frames']+max(room.frames for room in rooms)-1
    produced = 0
    pending = array('f')
    origin = 0

    def push(dry_part, wet_part, frames):
        nonlocal pending, produced
        frames = min(frames, total-produced)
        for local in range(frames):
            for channel in range(2):
                pending.append(dry_part[channel][local]*dry+wet_part[channel][local]*wet_gain)
        produced += frames
        while len(pending) >= block_frames*2:
            yield pending[:block_frames*2]
            del pending[:block_frames*2]

    while True:
        excitation, frames = _read_partition(source)
        if frames == 0:
            break
        table = _coeff_table(plan['stages'], origin, PARTITION)
        wet = [[0.0]*PARTITION for _ in range(2)]
        for index, room in enumerate(rooms):
            # A short final partition is zero-padded inside the input rows, so
            # the convolver still emits a full P-sample block whose extra
            # samples carry the reflections of the last real input samples;
            # the total-length cap trims anything beyond the convolution.
            rendered = room.advance(excitation, table[index])
            for channel in range(2):
                row, add = wet[channel], rendered[channel]
                for local in range(PARTITION):
                    row[local] += add[local]
        for piece in push(excitation, wet, PARTITION):
            yield piece
        origin += PARTITION

    # Drain every room until the full measured decay has been emitted.  Each
    # room keeps its own state, so reflections already in flight decay fully
    # instead of being cut off; finite responses guarantee the loop ends.
    silence = [[0.0]*PARTITION for _ in range(2)]
    while produced < total:
        wet = [[0.0]*PARTITION for _ in range(2)]
        for room in rooms:
            rendered = room.flush()
            for channel in range(2):
                row, add = wet[channel], rendered[channel]
                for local in range(PARTITION):
                    row[local] += add[local]
        for piece in push(silence, wet, PARTITION):
            yield piece
    if pending:
        yield pending
