#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include "hyper_ra.h"


void main(int argc, char** argv) {
    int ret;
    uint8_t quote_buf[2048] = { 0 };
    // Has to be 64 bytes user_data array to meet the length of sgx_report_data_t
    uint8_t user_data[64] = {1,2,3,4,5,6};

    if (argc != 2) {
        printf("CA json file has to be provided\n.");
        printf("Usage: hyper_ra_test <jsonfile>\n");
    }

    char *jsonfile = argv[1];

    printf("First, generate HyperEnclave RA quote.\n");
    ret = hyper_ra_gen(
        quote_buf, sizeof(quote_buf), user_data, sizeof(user_data)
    );
    if (ret != 0) {
        printf("hyper_ra_gen failed %d\n", ret);
        return;
    }

    printf("Successfully generated HyperEnclave RA quote.\n");
    printf("\nThen, verify HyperEnclave RA quote.\n");

    ret = hyper_ra_verify(
        quote_buf, sizeof(quote_buf), user_data, sizeof(user_data), jsonfile
    );
    if (ret != 0) {
        printf("hyper_ra_verify failed %d\n", ret);
        return;
    }
    printf("Verify successfully\n");
}