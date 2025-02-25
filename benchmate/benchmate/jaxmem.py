


def memory_peak_fetcher():
    import jax

    def fetch_memory_peak():
        # 'memory', 'memory_stats'
        devices = jax.devices()
        max_mem = -1
        for device in devices:
            # dqn.D0 [stdout] Device: cuda:0
            # dqn.D0 [stdout]   num_allocs: 0.0006799697875976562 MiB
            # dqn.D0 [stdout]   bytes_in_use: 0.915771484375 MiB
            # dqn.D0 [stdout]   peak_bytes_in_use: 80.41552734375 MiB
            # dqn.D0 [stdout]   largest_alloc_size: 16.07958984375 MiB
            # dqn.D0 [stdout]   bytes_limit: 60832.359375 MiB
            # dqn.D0 [stdout]   bytes_reserved: 0.0 MiB
            # dqn.D0 [stdout]   peak_bytes_reserved: 0.0 MiB
            # dqn.D0 [stdout]   largest_free_block_bytes: 0.0 MiB
            # dqn.D0 [stdout]   pool_bytes: 60832.359375 MiB
            # dqn.D0 [stdout]   peak_pool_bytes: 60832.359375 MiB
            # 
            # device_name = str(device)
            mem = device.memory_stats().get("peak_bytes_in_use", 0) / (1024 ** 2)
            max_mem = max(mem, max_mem)

        return max_mem
    
    return fetch_memory_peak
