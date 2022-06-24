#ifndef _HYPER_RA_TEST_H
#define _HYPER_RA_TEST_H

#ifdef __cplusplus
extern "C" {
#endif

int hyper_ra_gen(
        uint8_t *quote_buf,
        uint32_t quote_buf_len,
        uint8_t* user_data,
        uint32_t user_data_size
);

int hyper_ra_verify(uint8_t* quote_buf,
                    uint32_t quote_buf_len,
                    uint8_t* user_data,
                    uint32_t user_data_size,
                    const char *jsonfile
);

#ifdef __cplusplus
}
#endif

#endif  //_HYPER_RA_TEST_H