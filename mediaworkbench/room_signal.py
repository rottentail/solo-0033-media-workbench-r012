"""Partitioned causal matrix convolution on the programme excitation clock."""
from array import array
from collections import deque
from . import audio_signal, room_plan
from .denoise_signal import fft

PARTITION = 512


def weights(stages, sample):
    result = [0.0]*len(stages)
    active = 0
    while active+1<len(stages) and sample>=stages[active+1]['at']:
        active += 1
    stage = stages[active]
    if active and stage['fade'] and sample < stage['at']+stage['fade']:
        fraction = (sample-stage['at'])/stage['fade']
        result[active-1], result[active] = 1-fraction, fraction
    else:
        result[active] = 1.0
    return result


def _transfer(kernels, size):
    """Fold the transfer into the current processing window."""
    spectra = []
    for taps in kernels:
        window = [0.0]*size
        for index,value in enumerate(taps):
            window[index%size] += value
        spectra.append(fft(window))
    return spectra


def _window(source, frames, size):
    raw = audio_signal.read_frames(source,frames)
    if len(raw)!=frames*2:
        raise ValueError('programme spool truncated')
    channels = [[raw[2*i+c] for i in range(frames)]+[0.0]*(size-frames) for c in range(2)]
    return raw,[fft(channel) for channel in channels]


def blocks(source, plan, block_frames=4096):
    if type(block_frames) is not int or not 1<=block_frames<=65536:
        raise ValueError('invalid room output block size')
    size = 1
    while size<block_frames:
        size *= 2
    transfers = [_transfer(room_plan.kernels(s['response']),size) for s in plan['stages']]
    pending = array('f')
    for origin in range(0,plan['frames'],block_frames):
        count = min(block_frames,plan['frames']-origin)
        raw,channels = _window(source,count,size)
        coefficients = weights(plan['stages'],origin)
        rendered = []
        for transfer in transfers:
            room = []
            for out in range(2):
                bins = [channels[0][i]*transfer[out*2][i]+channels[1][i]*transfer[out*2+1][i] for i in range(size)]
                room.append(fft(bins,True))
            rendered.append(room)
        for local in range(count):
            for channel in range(2):
                wet = sum(rendered[j][channel][local].real*coefficient for j,coefficient in enumerate(coefficients))
                pending.append(raw[local*2+channel]*plan['dry']+wet*plan['wet'])
        if pending:
            yield pending
            pending = array('f')
