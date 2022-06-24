CC ?= gcc
LIBPATH = /opt/intel/sgxsdk/sdk_libs
INCPATH = -I /opt/intel/sgxsdk/include -I /usr/local/include
LINK_LIBS = -lsgx_uquote_verify_hyper -lcrypto -lcjson

CFLAGS = -fPIC -Wall -Wextra -O2 -g $(INCPATH)
LDFLAGS = -shared $(LINK_LIBS) -L $(LIBPATH)
TARGET_LIB = libhyper_ra.so

SRCS = $(wildcard *.c)
OBJS = $(SRCS:.c=.o)

.PHONY: all
all: ${TARGET_LIB}

$(TARGET_LIB): $(OBJS)
	$(CC) ${LDFLAGS} -o $@ $^

$(SRCS:.c=.d):%.d:%.c
	$(CC) $(CFLAGS) -MM $< >$@


.PHONY: clean
clean:
	rm -rf ${TARGET_LIB} ${OBJS} $(SRCS:.c=.d)
