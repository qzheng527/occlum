# Occlum Configuration

Occlum can be configured easily via a configuration file named `Occlum.json`, which is generated by the `occlum init` command in the Occlum instance directory. The user can modify `Occlum.json` to configure Occlum. The template of `Occlum.json` is shown below.

```json
{
    // Resource limits
    "resource_limits": {
        // The total size of enclave memory available to LibOS processes
        "user_space_size": "256MB",
        // The heap size of LibOS kernel
        "kernel_space_heap_size": "32MB",
        // The stack size of LibOS kernel
        "kernel_space_stack_size": "1MB",
        // The max number of LibOS threads/processes
        "max_num_of_threads": 32
    },
    // Process
    "process": {
        // The stack size of the "main" thread
        "default_stack_size": "4MB",
        // The max size of memory allocated by brk syscall
        "default_heap_size": "16MB",
        // The max size of memory by mmap syscall (OBSOLETE. Users don't need to modify this field. Keep it only for compatibility)
        "default_mmap_size": "32MB"
    },
    // Entry points
    //
    // Entry points specify all valid path prefixes for <path> in `occlum run
    // <path> <args>`. This prevents outside attackers from executing arbitrary
    // commands inside an Occlum-powered enclave.
    "entry_points": [
        "/bin"
    ],
    // Environment variables
    //
    // This gives a list of environment variables for the "root"
    // process started by `occlum exec` command.
    "env": {
        // The default env vars given to each "root" LibOS process. As these env vars
        // are specified in this config file, they are considered trusted.
        "default": [
            "OCCLUM=yes"
        ],
        // The untrusted env vars that are captured by Occlum from the host environment
        // and passed to the "root" LibOS processes. These untrusted env vars can
        // override the trusted, default envs specified above.
        "untrusted": [
            "EXAMPLE"
        ]
    },
    // Enclave metadata
    "metadata": {
        // Enclave signature structure's ISVPRODID field
        "product_id": 0,
        // Enclave signature structure's ISVSVN field
        "version_number": 0,
        // Whether the enclave is debuggable through special SGX instructions.
        // For production enclave, it is IMPORTANT to set this value to false.
        "debuggable": true,
        // Whether to turn on PKU feature in Occlum
        // Occlum uses PKU for isolation between LibOS and userspace program,
        // It is useful for developers to detect potential bugs.
        //
        // "pkru" = 0: PKU feature must be disabled
        // "pkru" = 1: PKU feature must be enabled
        // "pkru" = 2: PKU feature is enabled if the platform supports it
        "pkru": 0
    },
    // Mount points and their file systems
    //
    // The default configuration is shown below.
    "mount": [
        {
            "target": "/",
            "type": "unionfs",
            "options": {
                "layers": [
                    {
                        "target": "/",
                        "type": "sefs",
                        "source": "./build/mount/__ROOT",
                        "options": {
                            "MAC": ""
                        }
                    },
                    {
                        "target": "/",
                        "type": "sefs",
                        "source": "./run/mount/__ROOT"
                    }
                ]
            }
        },
        {
            "target": "/host",
            "type": "hostfs",
            "source": "."
        },
        {
            "target": "/proc",
            "type": "procfs"
        },
        {
            "target": "/dev",
            "type": "devfs"
        }
    ]
}
```

## Runtime Resource Configuration for Occlum process

Occlum has enabled per process resource configuration via [prlimit](https://man7.org/linux/man-pages//man2/prlimit.2.html) syscall and shell built-in command [ulimit](https://fishshell.com/docs/current/cmds/ulimit.html).

```shell
#! /usr/bin/bash
ulimit -a

# ulimit defined below will override configuration in Occlum.json
ulimit -Ss 10240 # stack size 10M
ulimit -Sd 40960 # heap size 40M
ulimit -Sv 102400 # virtual memory size 100M (including heap, stack, mmap size)

echo "ulimit result:"
ulimit -a

# Run applications with the new resource limits
...
```

For more info, please check [demos/fish](https://github.com/occlum/occlum/tree/master/demos/fish).