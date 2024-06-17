#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>

typedef struct {
    char *logfile;
    unsigned int interval_seconds;
} Args;

void read_meminfo(char *buffer, size_t bufsize) {
    FILE *file = fopen("/proc/meminfo", "r");
    if (file == NULL) {
        perror("Error opening /proc/meminfo");
        return;
    }

    size_t read_bytes = fread(buffer, 1, bufsize - 1, file);
    buffer[read_bytes] = '\0'; // null terminate the string

    fclose(file);
}

void write_to_log(const char *filename, const char *content) {
    FILE *file = fopen(filename, "a");
    if (file == NULL) {
        perror("Error opening log file");
        return;
    }

    fprintf(file, "%s\n", content);

    fclose(file);
}

void* timer_thread(void *arg) {
    Args *args = (Args*) arg;
    char buffer[4096];

    while (1) {
        sleep(args->interval_seconds);

        read_meminfo(buffer, sizeof(buffer));
        write_to_log(args->logfile, buffer);
    }

    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <logfile> <interval_seconds>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    Args args;
    args.logfile = argv[1];
    args.interval_seconds = atoi(argv[2]);

    pthread_t thread;
    if (pthread_create(&thread, NULL, timer_thread, &args) != 0) {
        perror("Error creating thread");
        exit(EXIT_FAILURE);
    }

    pthread_join(thread, NULL);

    return 0;
}