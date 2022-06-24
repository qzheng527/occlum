#ifndef _HELPER_H
#define _HELPER_H

#ifdef __cplusplus
extern "C" {
#endif

int gen_cert_from_json(const char *jsonfile, cert_chain_t* cert_chain);

#ifdef __cplusplus
}
#endif

#endif  //_HELPER_H