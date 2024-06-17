.PHONY: all clean

all: meminfo_logger

meminfo_logger: meminfo_logger.c
		$(CC) $^ -fPIE -pie -o $@ -pthread

clean:
		rm -rf meminfo_logger
