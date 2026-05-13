"""Subset pusht_expert_train.h5 to N random episodes (seed=42).

Reads source via Blosc (hdf5plugin), writes a new file with matching compression.
Streams one episode at a time to keep memory bounded.
"""
import os, sys, time
import hdf5plugin
import h5py
import numpy as np

SRC = "/home/qlib/.stable_worldmodel/pusht_expert_train.h5"
DST = "/home/qlib/.stable_worldmodel/pusht_expert_train.h5.new"
N_EPISODES = 500
SEED = 42

# Source filter parameters decoded from filter spec (32001, 1, (2,2,1,15052800,5,1,1), b'blosc')
# elem 4 = clevel=5, elem 5 = shuffle=1, elem 6 = compressor=1 (lz4)
BLOSC = hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)

def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    with h5py.File(SRC, "r") as src:
        ep_len = src["ep_len"][:]
        ep_offset = src["ep_offset"][:]
        n_eps_total = len(ep_len)
        assert n_eps_total == 18685, f"unexpected n_eps {n_eps_total}"

        # Random N episodes, sorted ascending so reads are sequential (much faster).
        chosen = np.sort(rng.choice(n_eps_total, size=N_EPISODES, replace=False))
        new_lens = ep_len[chosen]
        total_frames = int(new_lens.sum())
        print(f"Picked {N_EPISODES} episodes, total frames = {total_frames}")
        print(f"First 5 chosen ep ids: {chosen[:5]}, last: {chosen[-3:]}")

        # Build new ep_offset (cumulative)
        new_offsets = np.concatenate([[0], np.cumsum(new_lens[:-1])]).astype(np.int64)

        with h5py.File(DST, "w") as dst:
            # Match source dtypes from the actual datasets
            d_pixels = dst.create_dataset(
                "pixels",
                shape=(total_frames, 224, 224, 3),
                dtype="uint8",
                chunks=(100, 224, 224, 3),
                **BLOSC,
            )
            d_action = dst.create_dataset("action", (total_frames, 2), dtype="float32", chunks=(1000, 2))
            d_proprio = dst.create_dataset("proprio", (total_frames, 4), dtype="float32", chunks=(1000, 4))
            d_state = dst.create_dataset("state", (total_frames, 7), dtype="float32", chunks=(1000, 7))
            d_epidx = dst.create_dataset("episode_idx", (total_frames,), dtype="int64", chunks=(1000,))
            d_stepidx = dst.create_dataset("step_idx", (total_frames,), dtype="int64", chunks=(1000,))
            dst.create_dataset("ep_len", data=new_lens.astype(np.int32))
            dst.create_dataset("ep_offset", data=new_offsets)

            src_pix = src["pixels"]
            src_act = src["action"]
            src_pro = src["proprio"]
            src_st = src["state"]
            src_step = src["step_idx"]

            cursor = 0
            t_last = time.time()
            for new_id, src_id in enumerate(chosen):
                start = int(ep_offset[src_id])
                length = int(ep_len[src_id])
                end = start + length

                d_pixels[cursor:cursor + length] = src_pix[start:end]
                d_action[cursor:cursor + length] = src_act[start:end]
                d_proprio[cursor:cursor + length] = src_pro[start:end]
                d_state[cursor:cursor + length] = src_st[start:end]
                # Remap episode_idx to new dense ids 0..N-1
                d_epidx[cursor:cursor + length] = new_id
                # Preserve step_idx (within-episode positions are unchanged)
                d_stepidx[cursor:cursor + length] = src_step[start:end]

                cursor += length
                if (new_id + 1) % 25 == 0:
                    now = time.time()
                    rate = 25 / (now - t_last) if now > t_last else 0
                    eta = (N_EPISODES - new_id - 1) / rate if rate > 0 else 0
                    print(f"  [{new_id+1}/{N_EPISODES}] frames={cursor}/{total_frames} "
                          f"rate={rate:.1f} ep/s eta={eta:.0f}s "
                          f"file_size={os.path.getsize(DST)/1e9:.2f}G",
                          flush=True)
                    t_last = now

            assert cursor == total_frames, f"cursor {cursor} != total {total_frames}"

    elapsed = time.time() - t0
    final_size = os.path.getsize(DST)
    src_size = os.path.getsize(SRC)
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  src: {src_size/1e9:.2f} GB")
    print(f"  dst: {final_size/1e9:.2f} GB ({100*final_size/src_size:.2f}% of src)")

if __name__ == "__main__":
    main()
