# TEE Memory Status Check

Sometimes it is necessary to check the status of memory in a trusted execution environment (TEE) to assist memory configuration and application operation.
This can be done by reading the node **/proc/meminfo** in Occlum starting from Occlum version 0.31.0.

For exmaple, you may get below information by reading **/proc/meminfo**.
```bash
MemTotal:              307200 kB
MemFree:               130372 kB
MemAvailable:          130372 kB
KernelHeapTotal:       32768 kB
KernelHeapPeakUsed:    15344 kB
KernelHeapInUse:       9036 kB
```

The above information shows that there are 307200 KB of total TEE memory, 130372 KB of free memory, and 130372 KB of available memory. The Kernel Heap Total size is 32768 KB, and the peak used size is 15344 KB. The current usage size is 9036 KB.

To assist the memory status check, an application [meminfo_logger](./meminfo_logger.c) is provided as example.
This application reads the node **/proc/meminfo** in a configurable inteval and write the memory status to a file.

You can run the application with command below.
```bash
$ occlum run /bin/meminfo_logger  <log_file> <interval>
```

Build to occlum instance.
```bash
$ ./build.sh
```

Run the memory status check every 5 seconds with logfile **/host/meminfo.log** which could be found in directory **occlum_instance** in default.
```bash
$ occlum run /bin/meminfo_logger /host/meminfo.log 5
```
