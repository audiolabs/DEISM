# Pyenv + Conda Interference Troubleshooting Guide

## Problem
When using both pyenv and conda, pyenv can intercept Python commands even when conda environments are active, causing architecture mismatches (e.g., x86_64 vs ARM64).

## Detection Methods

### 1. Quick Detection
```bash
# Check if pyenv is intercepting python command
which python
# If it returns something like: /Users/username/.pyenv/shims/python
# Then pyenv is intercepting the command

# Check the actual conda Python
conda activate your_env
ls -la $CONDA_PREFIX/bin/python*
```

### 2. Architecture Check
```bash
# Check what architecture Python is actually running
python -c "import platform; print('Architecture:', platform.machine())"

# Check conda's Python architecture directly
$CONDA_PREFIX/bin/python -c "import platform; print('Architecture:', platform.machine())"
```

### 3. PATH Analysis
```bash
# Check PATH order
echo $PATH | tr ':' '\n' | head -5
# Look for: /Users/username/.pyenv/shims should NOT be first when conda is active
```

## Solutions

### Solution 1: Use Full Path to Conda Python (Recommended)
```bash
# Instead of: python setup.py build_ext --inplace
# Use: $CONDA_PREFIX/bin/python setup.py build_ext --inplace

# Or for your specific case:
/Users/xuzeyu/miniconda3/envs/DEISM_github_dev/bin/python3.9 setup.py build_ext --inplace
```

### Solution 2: Temporarily Modify PATH
```bash
# Remove pyenv from PATH temporarily
export PATH=$(echo $PATH | sed 's|/Users/username/.pyenv/shims:||g')
python setup.py build_ext --inplace
```

### Solution 3: Temporarily Disable Pyenv
```bash
# Disable pyenv for current session
unset PYENV_ROOT
export PATH=$(echo $PATH | sed 's|/Users/username/.pyenv/shims:||g')
python setup.py build_ext --inplace
```

### Solution 4: Use Conda's Python Explicitly
```bash
# Set alias for current session
alias python="$CONDA_PREFIX/bin/python"
python setup.py build_ext --inplace
```

## Permanent Solutions

### Option A: Modify Shell Configuration
Add to your `~/.zshrc` or `~/.bashrc`:
```bash
# Put conda bin before pyenv when conda is active
if [[ -n "$CONDA_PREFIX" ]]; then
    export PATH="$CONDA_PREFIX/bin:$PATH"
fi
```

### Option B: Use Conda's Python Manager
```bash
# Use conda to manage Python versions instead of pyenv
conda install python=3.9  # or whatever version you need
```

## Verification Commands

### Check if Solution Worked
```bash
# After applying solution, verify:
which python  # Should point to conda's Python
python -c "import platform; print('Architecture:', platform.machine())"  # Should match your system
```

### For Your Specific Case (Apple Silicon)
```bash
# Should return 'arm64' for native Apple Silicon
python -c "import platform; print('Architecture:', platform.machine())"
```

## Common Symptoms

1. **Architecture Mismatch Error:**
   ```
   ImportError: dlopen(...): mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64e' or 'arm64')
   ```

2. **Wrong Python Path:**
   ```bash
   which python
   # Returns: /Users/username/.pyenv/shims/python (should be conda's path)
   ```

3. **Build Using Wrong Architecture:**
   - Compiler shows `-arch x86_64` instead of `-arch arm64`
   - No "Detected Apple Silicon hardware" message during build

## Prevention

1. **Check Before Building:**
   ```bash
   conda activate your_env
   which python  # Should NOT contain .pyenv/shims
   python -c "import platform; print(platform.machine())"  # Should match your system
   ```

2. **Use Conda's Python Explicitly:**
   ```bash
   # Always use full path when building extensions
   $CONDA_PREFIX/bin/python setup.py build_ext --inplace
   ```

3. **Environment Isolation:**
   ```bash
   # Create separate conda environments for different projects
   conda create -n project_name python=3.9
   conda activate project_name
   ```

## Debugging Commands

```bash
# Full diagnostic
conda activate your_env
echo "=== Python Path ==="
which python
echo "=== Python Architecture ==="
python -c "import platform; print('Arch:', platform.machine(), 'Platform:', platform.platform())"
echo "=== Conda Python Architecture ==="
$CONDA_PREFIX/bin/python -c "import platform; print('Arch:', platform.machine())"
echo "=== PATH Order ==="
echo $PATH | tr ':' '\n' | head -5
echo "=== Pyenv Status ==="
pyenv version 2>/dev/null || echo "Pyenv not active"
```

## Your Specific Fix for DEISM

For your current DEISM project, use:
```bash
cd /Users/xuzeyu/Desktop/Projects/DEISM/Github/DEISM
conda activate DEISM_github_dev
/Users/xuzeyu/miniconda3/envs/DEISM_github_dev/bin/python3.9 setup.py build_ext --inplace
```

This ensures you're using conda's native ARM64 Python instead of pyenv's x86_64 Python.
