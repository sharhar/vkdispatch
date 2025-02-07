from packaging import version
import re

def parse_compiler_version(version_output):
    if not isinstance(version_output, str):
        return None
    
    # Try to match either clang or gcc version string
    clang_match = re.search(r'clang version ([^\s]+)', version_output)
    gcc_match = re.search(r'gcc.+?([\d.]+(?:-[a-zA-Z0-9]+)?)', version_output, re.IGNORECASE)
    
    match = clang_match or gcc_match
    if not match:
        return None
    
    print(f"Match: {match.group(0)}")
    print(f"Version: {match.group(1)}")

    try:
        return version.parse(match.group(1))
    except version.InvalidVersion as e:
        print(f"Invalid version: {e}")
        return None

# Example usage:
def test_versions():
    test_cases = [
        "clang version 14.0.6",
        "gcc version 12.2.0 (GCC)",
        "Apple clang version 14.0.3",
        "gcc (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0",
        "This is not a version string",
        None,
        "gcc version invalid",
        """
Apple clang version 15.0.0 (clang-1500.0.40.1)
Target: arm64-apple-darwin23.1.0
Thread model: posix
""",
"""
gcc (GCC) 8.3.1 20191121 (Red Hat 8.3.1-5)
Copyright (C) 2018 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
""",
"""
gcc (GCC) ((8.3.1) 20191121 (Red Hat 8.3.1-5)
Copyright (C) 2018 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
"""
    ]
    
    for test in test_cases:
        ver = parse_compiler_version(test)
        if ver:
            print(f"Input: {test}\nParsed version: {ver}\n")
        else:
            print(f"Input: {test}\nNo version found\n")

test_versions()
