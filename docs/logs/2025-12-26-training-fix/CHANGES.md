# Code Changes

## src/ckernel_codegen.c

### Change 1: Add environment variable check (lines 820-825)

```diff
     "    g_model.num_cores = 1;\n"
     "    g_model.task_type = TASK_LM;\n"
+    "    /* Check env var to pre-allocate gradient buffers for training */\n"
+    "    const char *train_env = getenv(\"CK_ENABLE_TRAINING\");\n"
+    "    if (train_env && (train_env[0] == '1' || train_env[0] == 'y' || train_env[0] == 'Y')) {\n"
+    "        g_model.training_enabled = true;\n"
+    "        g_model.learning_rate = 1e-4f;\n"
+    "    }\n"
     "    if (layout_model(&g_model) != 0) return -1;\n"
```

### Change 2: Remove duplicate SGD from backward (line 1268)

```diff
             "                           m->rope_theta <= 0.0f);\n"
             "    }\n"
             "\n"
-            "    sgd_update(m, m->learning_rate);\n"
+            "    /* SGD update is now called separately via optimizer_step() */\n"
             "    return 0;\n"
             "}\n\n");
```

## New Files

### scripts/test_training_fix.sh

Complete test script that:
1. Builds everything from clean
2. Regenerates model.c
3. Verifies fixes are present
4. Recompiles libmodel.so
5. Runs training test with CK_ENABLE_TRAINING=1
