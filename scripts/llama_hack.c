if (state->ith == 0 && node->name[0] != 0) {
            printf("[NODE] %s\n", node->name);
            if (strstr(node->name, "blk.0") || strstr(node->name, "output") || strstr(node->name, "result")) {
                struct stat st = {0};
                if (stat("llama_dump", &st) == -1) { mkdir("llama_dump", 0700); }
                char fname[512];
                snprintf(fname, sizeof(fname), "llama_dump/%s.bin", node->name);
                FILE *f = fopen(fname, "wb");
                if (f) {
                    fwrite(node->data, 1, ggml_nbytes(node), f);
                    fclose(f);
                    printf("[DUMP] %s\n", node->name);
                }
            }
        }