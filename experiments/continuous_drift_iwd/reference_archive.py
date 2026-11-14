"""Retain the generated reference for a single uninterrupted evaluation."""
import numpy as np


class ReferenceArchive:
    def __init__(self, env):
        self.streaming = bool(getattr(env, '_streaming_reference', False))
        self.parts = []
        self.offsets = []
        self.times = []
        self.end = None
        self.capture(env, 0.)

    def capture(self, env, time):
        offset = int(env._rolling_reference.offset[0]) if self.streaming else 0
        if self.offsets and offset == self.offsets[-1]:
            return
        if self.offsets and offset < self.offsets[-1]:
            raise ValueError('Reference archive cannot span an episode reset')
        road = env.path_bank[env.path_generation[0], env.path_id[0]].detach().cpu().numpy()
        if self.end is not None and offset > self.end:
            raise ValueError('Reference archive missed a complete reference window')
        start = 0 if self.end is None else self.end - offset
        if start < len(road):
            self.parts.append(road[start:].copy())
        self.end = offset + len(road)
        self.offsets.append(offset)
        self.times.append(float(time))

    def save(self, path, metadata_path):
        np.save(path, np.concatenate(self.parts, axis=0))
        np.savez(metadata_path, window_start_index=np.asarray(self.offsets),
                 window_start_time=np.asarray(self.times), streaming=self.streaming)
