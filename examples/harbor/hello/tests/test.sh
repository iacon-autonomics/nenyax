#!/bin/bash
if [ "$(cat /app/hello.txt 2>/dev/null | tr -d '\n')" = "hi" ]; then echo 1 > /logs/verifier/reward.txt; echo PASS; else echo 0 > /logs/verifier/reward.txt; echo FAIL; fi
