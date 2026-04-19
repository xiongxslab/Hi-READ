#!/usr/bin/env python3
"""
Test script to validate the main public Hi-READ workflow entry points.

This script performs basic validation checks on the main workflow entry points:
1. Loop benchmark wrapper
2. K-means clustering pipeline
3. SV-gene-disease rebuild step

It checks for:
- File existence
- Import capability
- Argument parser functionality
- Basic syntax validation
"""

import sys
import subprocess
import importlib.util
from pathlib import Path

def test_file_exists(filepath):
    """Check if file exists"""
    if filepath.exists():
        print(f"  ✓ File exists: {filepath.name}")
        return True
    else:
        print(f"  ✗ File missing: {filepath}")
        return False

def test_import(filepath):
    """Test if Python file can be imported"""
    try:
        sys.path.insert(0, str(filepath.parent))
        spec = importlib.util.spec_from_file_location("test_module", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"  ✓ Import successful: {filepath.name}")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {filepath.name}")
        print(f"    Error: {str(e)[:100]}")
        return False

def test_shell_syntax(filepath):
    """Check if a shell script parses cleanly."""
    try:
        result = subprocess.run(
            ["bash", "-n", str(filepath)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            print(f"  ✓ Shell syntax valid: {filepath.name}")
            return True
        print(f"  ✗ Shell syntax failed: {filepath.name}")
        print(f"    Error: {(result.stderr or result.stdout).strip()[:100]}")
        return False
    except Exception as e:
        print(f"  ✗ Shell syntax check failed: {filepath.name}")
        print(f"    Error: {str(e)[:100]}")
        return False

def test_has_main(filepath):
    """Check if file has main() function or __main__ block"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        has_main_func = 'def main(' in content
        has_main_block = "if __name__ == '__main__':" in content or 'if __name__ == "__main__":' in content

        if has_main_func or has_main_block:
            print(f"  ✓ Has main entry point: {filepath.name}")
            return True
        else:
            print(f"  ✗ No main entry point: {filepath.name}")
            return False
    except Exception as e:
        print(f"  ✗ Error reading file: {str(e)[:100]}")
        return False

def test_has_argparse(filepath):
    """Check if file uses argparse"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        if 'argparse' in content or 'ArgumentParser' in content:
            print(f"  ✓ Uses argparse: {filepath.name}")
            return True
        else:
            print(f"  ⚠ No argparse found: {filepath.name}")
            return False
    except Exception as e:
        print(f"  ✗ Error reading file: {str(e)[:100]}")
        return False

def test_no_chinese(filepath):
    """Check if file contains Chinese characters"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for common Chinese characters
        chinese_chars = [c for c in content if '\u4e00' <= c <= '\u9fff']

        if len(chinese_chars) == 0:
            print(f"  ✓ No Chinese characters: {filepath.name}")
            return True
        else:
            print(f"  ✗ Contains {len(chinese_chars)} Chinese characters: {filepath.name}")
            return False
    except Exception as e:
        print(f"  ✗ Error reading file: {str(e)[:100]}")
        return False

def test_pipeline(name, filepath):
    """Run all tests for a pipeline"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    results = []

    # Test 1: File exists
    results.append(test_file_exists(filepath))

    if not results[0]:
        print(f"\n✗ {name}: FAILED (file not found)")
        return False

    # Test 2: No Chinese characters
    results.append(test_no_chinese(filepath))

    # Test 3: Has main entry point
    results.append(test_has_main(filepath))

    # Test 4: Uses argparse
    results.append(test_has_argparse(filepath))

    # Test 5: Can import or parse (syntax check)
    if filepath.suffix == '.py':
        results.append(test_import(filepath))
    elif filepath.suffix == '.sh':
        results.append(test_shell_syntax(filepath))
    else:
        print(f"  ⚠ No syntax checker for: {filepath.name}")
        results.append(False)

    # Summary
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"\n✓ {name}: PASSED ({passed}/{total} checks)")
        return True
    else:
        print(f"\n⚠ {name}: PARTIAL ({passed}/{total} checks passed)")
        return False

def main():
    """Main test runner"""
    print("="*60)
    print("Hi-READ Pipeline Validation Tests")
    print("="*60)

    # Define pipeline paths
    base_dir = Path(__file__).parent

    pipelines = [
        ("Loop Benchmark", base_dir / "workflows" / "loop_benchmark" / "run_loop_benchmark.py"),
        ("Loops Cluster", base_dir / "workflows" / "loops_cluster" / "code" / "run_cluster_only_pipeline.py"),
        ("SV-Gene-Disease Rebuild", base_dir / "workflows" / "sv_gene_disease_analysis" / "scripts" / "step1_rebuild_analysis_tables.py"),
    ]

    results = []

    # Test each pipeline
    for name, filepath in pipelines:
        result = test_pipeline(name, filepath)
        results.append((name, result))

    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\nTotal: {passed}/{total} pipelines passed")

    if passed == total:
        print("\n✓ All pipelines validated successfully!")
        return 0
    else:
        print(f"\n⚠ {total - passed} pipeline(s) need attention")
        return 1

if __name__ == '__main__':
    sys.exit(main())
