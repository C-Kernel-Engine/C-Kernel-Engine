import unittest

from scripts.check_openmp_policy import violations


class OpenMPPolicyTest(unittest.TestCase):
    def test_native_and_build_opt_ins(self):
        for code in ("#pragma omp parallel for", "#include <omp.h>", "#ifdef _OPENMP",
                     "lib.omp_set_num_threads(2)", "openmp=args.threads > 1",
                     "options += ['-fopenmp']", "target_link_libraries(x OpenMP::OpenMP_C)"):
            with self.subTest(code=code):
                self.assertEqual(list(violations("+++ b/a.c\n@@ -0,0 +8 @@\n+" + code)),
                                 ["a.c:8: " + code])

    def test_migration_can_remove_openmp(self):
        diff = "+++ b/a.c\n@@ -1 +1 @@\n-#pragma omp parallel for\n+ck_threadpool_dispatch(pool, fn, args);"
        self.assertEqual(list(violations(diff)), [])
