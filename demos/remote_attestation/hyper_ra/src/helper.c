#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

#include <cjson/cJSON.h>
#include "sgx_quote_verify.h"


static cJSON *read_to_cjson(const char *jsonfile)
{
    struct stat statbuf = { 0 };
    int fileSize;
    if (stat(jsonfile, &statbuf) == 0) {
        fileSize = statbuf.st_size;
    } else {
        printf("Get json file %s size failed.", jsonfile);
        return NULL;
    }

    // Allocate memory for json string
    char *jsonStr = (char *)malloc(sizeof(char) * fileSize + 1);
    memset(jsonStr, 0, fileSize + 1);

    FILE *file = NULL;
    file = fopen(jsonfile, "r");
    if (file == NULL) {
        printf("Open json file %s failed.", jsonfile);
        free(jsonStr);
        return NULL;
    }

    // Read json string in file
    int size = fread(jsonStr, sizeof(char), fileSize, file);
    if (size == 0) {
        printf("Failed to read json file %s.", jsonfile);
        free(jsonStr);
        fclose(file);
        return NULL;
    }
    // printf("%s", jsonStr);
    fclose(file);

    // Convert the read json string into a json variable pointer
    cJSON *root = cJSON_Parse(jsonStr);
    if (!root) {
        const char *err = cJSON_GetErrorPtr();
        printf("Error before: [%s]", err);
        free((void *)err);
        free(jsonStr);
        return NULL;
    }
    free(jsonStr);
    return root;
}

static int cjson_get_int(cJSON *root, char *name)
{
    cJSON *item = NULL;
    item = cJSON_GetObjectItem(root, name);	
    if (item != NULL && cJSON_IsNumber(item)) {
        return item->valueint;
    } else {
        printf("not found valid int for %s.\n", name);
        return -1;
    }
}

static char *cjson_get_string(cJSON *root, char *name)
{
    cJSON *item = NULL;
    item = cJSON_GetObjectItem(root, name);	
    if (item != NULL && cJSON_IsString(item) && item->valuestring != NULL) {
        return item->valuestring;
    } else {
        printf("not found valid string for %s.\n", name);
        return NULL;
    }
}

int gen_cert_from_json(const char *jsonfile, cert_chain_t* cert_chain)
{
    cJSON *root = read_to_cjson(jsonfile);
    if (!root)
        return -1;

    int ret = 0;
    int cert_mode = cjson_get_int(root, "cert_mode");
    cert_chain->cert_mode = cert_mode;
    if(cert_mode == CFCA_CERT) {
        char *cfca_acs_sm2_ca = cjson_get_string(root, "cfca_acs_sm2_ca");
        if (!cfca_acs_sm2_ca) {
            ret = -1;
            goto exit;
        }

        char *cfca_acs_sm2_oca33 = cjson_get_string(root, "cfca_acs_sm2_oca33");
        if (!cfca_acs_sm2_oca33) {
            ret = -1;
            goto exit;
        }

        uint32_t cfca_acs_sm2_ca_len = (uint32_t)strlen(cfca_acs_sm2_ca);
        memcpy(cert_chain->cfca_cert.root_cert.pem, cfca_acs_sm2_ca, cfca_acs_sm2_ca_len);
        cert_chain->cfca_cert.root_cert.pem_len = cfca_acs_sm2_ca_len;
        uint32_t cfca_acs_sm2_oca33_len = (uint32_t)strlen(cfca_acs_sm2_oca33);
        memcpy(cert_chain->cfca_cert.second_level_cert.pem, cfca_acs_sm2_oca33, cfca_acs_sm2_oca33_len);
        cert_chain->cfca_cert.second_level_cert.pem_len = cfca_acs_sm2_oca33_len;
    } else if (cert_mode == OTHER_CA_CERT) {
        char *other_ca = cjson_get_string(root, "other_ca");
        if (!other_ca) {
            ret = -1;
            goto exit;
        }
        uint32_t other_ca_len = (uint32_t)strlen(other_ca);
        memcpy(cert_chain->other_ca_cert.root_cert.pem, other_ca, other_ca_len);
        cert_chain->other_ca_cert.root_cert.pem_len = other_ca_len;
    } else if (cert_mode == NO_CERT) {
        char *tpm_ak_pub_baseline = cjson_get_string(root, "tpm_ak_pub_baseline");
        if (!tpm_ak_pub_baseline) {
            ret = -1;
            goto exit;
        }
        char *pcr_general_baseline = cjson_get_string(root, "pcr_general_baseline");
        if (!pcr_general_baseline) {
            ret = -1;
            goto exit;
        }
        char *pcr_5_baseline = cjson_get_string(root, "pcr_5_baseline");
        if (!pcr_5_baseline) {
            ret = -1;
            goto exit;
        }
        char *pcr_13_baseline = cjson_get_string(root, "pcr_13_baseline");
        if (!pcr_13_baseline) {
            ret = -1;
            goto exit;
        }
        // printf("tpm_ak_pub_baseline: %s\n", tpm_ak_pub_baseline);
        // printf("pcr_general_baseline: %s\n", pcr_general_baseline);
        // printf("pcr_5_baseline: %s\n", pcr_5_baseline);
        // printf("pcr_13_baseline: %s\n", pcr_13_baseline);
        memcpy(cert_chain->no_cert.raw_config.tpm_ak_pub, tpm_ak_pub_baseline, strlen(tpm_ak_pub_baseline));
        memcpy(cert_chain->no_cert.raw_config.pcr_general, pcr_general_baseline, strlen(pcr_general_baseline));
        memcpy(cert_chain->no_cert.raw_config.pcr_5, pcr_5_baseline, strlen(pcr_5_baseline));
        memcpy(cert_chain->no_cert.raw_config.pcr_13, pcr_13_baseline, strlen(pcr_13_baseline));
    } else {
        printf("Cert mode %d is not valid.\n", cert_mode);
        ret = -1;
        goto exit;
    }

exit:
    cJSON_Delete(root);
    return ret;
}