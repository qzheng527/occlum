CC ?= gcc
LD ?= ld
LIBPATH ?= ../src
# INCPATH ?= -I /opt/intel/sgxsdk/include
# INCPATH += -I /usr/local/include
# LINK_FLAGS ?= -lsgx_uquote_verify_hyper
# LINK_FLAGS += -lcrypto -lcjson
INCPATH = -I ../src
LINK_LIBS = -lhyper_ra

.PHONY: all clean

all: ra_test

ra_test: ra_test.c
	$(CC) $^ -fPIE -pie -o $@ -L $(LIBPATH) $(INCPATH) $(LINK_LIBS)

clean:
	rm -rf ra_test
