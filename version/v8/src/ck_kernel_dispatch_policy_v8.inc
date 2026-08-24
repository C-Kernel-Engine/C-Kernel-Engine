/* Generated from version/v8/kernel_maps. Do not edit manually. */
/* Generator: version/v8/scripts/generate_kernel_dispatch_policy_v8.py */

static const ck_gemm_route_v8 ck_policy_gemm_nt_q4_k_q8_k_avx512_vnni_x16_decode_prepared[] = {
    {1, 1, 6144, 6144, 5120, 5120, 0, 0, 0, 0},
    {1, 1, 12288, 12288, 5120, 5120, 0, 0, 0, 0},
    {1, 1, 17408, 17408, 5120, 5120, 0, 0, 0, 0},
    {1, 1, 5120, 5120, 6144, 6144, 0, 0, 0, 0},
    {1, 1, 5120, 5120, 17408, 17408, 0, 0, 0, 0},
};
#define CK_POLICY_GEMM_NT_Q4_K_Q8_K_AVX512_VNNI_X16_DECODE_PREPARED_COUNT (sizeof(ck_policy_gemm_nt_q4_k_q8_k_avx512_vnni_x16_decode_prepared) / sizeof(ck_policy_gemm_nt_q4_k_q8_k_avx512_vnni_x16_decode_prepared[0]))

static const ck_gemm_route_v8 ck_policy_gemm_nt_q4_k_q8_k_avx512_vnni_x16_prefill[] = {
    {16, 2147483647, 1024, 1024, 5120, 5120, 0, 0, 0, 0},
    {16, 2147483647, 6144, 6144, 5120, 5120, 0, 0, 0, CK_GEMM_ROUTE_BATCHED_TAIL},
    {16, 2147483647, 12288, 12288, 5120, 5120, 0, 0, 0, 0},
    {16, 2147483647, 34816, 34816, 5120, 5120, 0, 0, 20, 0},
    {16, 2147483647, 5120, 5120, 6144, 6144, 0, 0, 0, 0},
    {16, 2147483647, 5120, 5120, 17408, 17408, 0, 0, 0, 0},
};
#define CK_POLICY_GEMM_NT_Q4_K_Q8_K_AVX512_VNNI_X16_PREFILL_COUNT (sizeof(ck_policy_gemm_nt_q4_k_q8_k_avx512_vnni_x16_prefill) / sizeof(ck_policy_gemm_nt_q4_k_q8_k_avx512_vnni_x16_prefill[0]))

static const ck_gemm_route_v8 ck_policy_gemm_nt_q6_k_q8_k_prefill_schedule[] = {
    {4, 63, 10240, 10240, 5120, 5120, 16, 64, 0, CK_GEMM_ROUTE_OUTPUT_TILES | CK_GEMM_ROUTE_COMPACT_M4},
    {4, 63, 4096, 2147483647, 8192, 2147483647, 8, 128, 0, CK_GEMM_ROUTE_OUTPUT_TILES | CK_GEMM_ROUTE_COMPACT_M4},
    {2, 3, 4096, 2147483647, 8192, 2147483647, 8, 128, 0, CK_GEMM_ROUTE_OUTPUT_TILES},
    {64, 2147483647, 1024, 2147483647, 5120, 2147483647, 16, 256, 0, CK_GEMM_ROUTE_OUTPUT_TILES | CK_GEMM_ROUTE_COMPACT_M4},
    {2, 2147483647, 2048, 2147483647, 8192, 2147483647, 16, 256, 0, CK_GEMM_ROUTE_OUTPUT_TILES},
    {1, 2147483647, 1, 2147483647, 1, 2147483647, 16, 256, 0, 0},
};
#define CK_POLICY_GEMM_NT_Q6_K_Q8_K_PREFILL_SCHEDULE_COUNT (sizeof(ck_policy_gemm_nt_q6_k_q8_k_prefill_schedule) / sizeof(ck_policy_gemm_nt_q6_k_q8_k_prefill_schedule[0]))
