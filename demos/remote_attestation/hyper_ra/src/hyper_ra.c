#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sgx_report.h>
#include <sgx_quote.h>

#include "sgx_quote_verify.h"
#include "helper.h"


#define IOC_HYPER_GEN_RA_QUOTE      _IOWR('s', 2, hyper_ra_gen_quote_arg_t)

typedef struct {
    sgx_report_data_t           report_data;        // input
    sgx_quote_sign_type_t       quote_type;         // input
    sgx_spid_t                  spid;               // input
    sgx_quote_nonce_t           nonce;              // input
    const uint8_t              *sigrl_ptr;          // input (optional)
    uint32_t                    sigrl_len;          // input (optional)
    uint32_t                    quote_buf_len;      // input
    union {
        uint8_t                *as_buf;
        sgx_quote_t            *as_quote;
    } quote;                                        // output
} hyper_ra_gen_quote_arg_t;


int hyper_ra_gen(
        uint8_t *quote_buf,
        uint32_t quote_buf_len,
        uint8_t* user_data,
        uint32_t user_data_size
        )
{
    int ret;
    int sgx_fd;
    if ((sgx_fd = open("/dev/sgx", O_RDONLY)) < 0) {
        printf("failed to open /dev/sgx\n");
        return -1;
    }

    if (!quote_buf || !user_data || !quote_buf_len || !user_data_size) {
        printf("Invalid parameters\n");
        return -1;
    }

    if (user_data_size > SGX_REPORT_DATA_SIZE) {
        printf("User data length is longer than %d\n", SGX_REPORT_DATA_SIZE);
        return -1;
    }

    hyper_ra_gen_quote_arg_t gen_quote_arg = {
        .report_data = { { 0 } },                       // input (empty is ok)
        .quote_type = SGX_LINKABLE_SIGNATURE,           // input
        .spid = { { 0 } },                              // input (empty is ok)
        .nonce = { { 0 } },                             // input (empty is ok)
        .sigrl_ptr = NULL,                              // input (optional)
        .sigrl_len = 0,                                 // input (optional)
        .quote_buf_len = quote_buf_len,             // input
        .quote = { .as_buf = (uint8_t *) quote_buf }    // output
    };

    // fill in the report data buffer
    memcpy(&gen_quote_arg.report_data, user_data, user_data_size);

    ret = ioctl(sgx_fd, IOC_HYPER_GEN_RA_QUOTE, &gen_quote_arg);
    if (ret != 0) {
        printf("failed to ioctl /dev/sgx, return %d\n", ret);
    }

    // Do simple check
    sgx_quote_t *quote = gen_quote_arg.quote.as_quote;

    if (quote->signature_len == 0) {
        printf("invalid quote: zero-length signature\n");
        ret = -1;
    }
    if (memcmp(&gen_quote_arg.report_data, &quote->report_body.report_data,
               sizeof(sgx_report_data_t)) != 0) {
        printf("invalid quote: wrong report data\n");
        ret = -1;
    }

    close(sgx_fd);
    return ret;
}

int hyper_ra_verify(uint8_t* quote_buf,
                    uint32_t quote_buf_len,
                    uint8_t* user_data,
                    uint32_t user_data_size,
                    const char *jsonfile
                    )
{
    int ret;
    cert_chain_t cert_chain = {0};

    if (!quote_buf || !quote_buf_len ) {
        printf("Invalid quote_buf or quote_buf_len.\n");
        return -1;
    }

    if (user_data && user_data_size != SGX_REPORT_DATA_SIZE) {
        printf("User data or user_data_size are not valid.\n");
        return -1;
    }

    if (gen_cert_from_json(jsonfile, &cert_chain)) {
        printf("Error, get root cert file fail\n.");
        return -1;
    }

    ret = sgx_verify_quote(
        quote_buf, quote_buf_len, user_data, user_data_size, &cert_chain);
    if (ret != SGX_QV_SUCCESS) {
        printf("Error, sgx_verify_quote fail in 0x[%04x].\n", ret);
        return -1;
    }

    return 0;
}
