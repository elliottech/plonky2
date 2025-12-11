# NVIDIA Driver Fix After Reboot

## Problem
After system reboot, `nvidia-smi` fails with error:
```
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

## Root Cause
System updates install new kernel versions, but NVIDIA driver modules aren't automatically built for the new kernel because kernel headers are missing.

## Quick Fix (Run after each kernel update)

```bash
# 1. Install kernel headers for current kernel
sudo apt update
sudo apt install -y linux-headers-$(uname -r)

# 2. DKMS will automatically rebuild NVIDIA modules
# Wait for the installation to complete (shows "Building module(s)..." and "done")

# 3. Load the NVIDIA driver
sudo modprobe nvidia

# 4. Verify it works
nvidia-smi
```

## Diagnostic Commands

Check if NVIDIA modules are built for current kernel:
```bash
uname -r                    # Show current kernel version
dkms status                 # Check which kernels have NVIDIA modules
modinfo nvidia              # Check if nvidia module exists for current kernel
lsmod | grep nvidia         # Check if nvidia modules are loaded
```

## Prevention - Auto-install Headers (Recommended)

Set up automatic kernel header installation:
```bash
sudo bash -c 'cat > /etc/apt/apt.conf.d/99auto-kernel-headers <<EOF
# Automatically install kernel headers when kernel is upgraded
DPkg::Post-Invoke {
  "if [ -x /usr/bin/dkms ]; then /usr/bin/apt-get install -y linux-headers-\$(uname -r) || true; fi";
};
EOF'
```

After this one-time setup, kernel headers will be installed automatically with kernel updates.

## Alternative - Pin Kernel Version (Not Recommended)

If you want to prevent kernel updates entirely:
```bash
# Hold current kernel version
sudo apt-mark hold linux-image-aws linux-headers-aws

# To allow updates again later
sudo apt-mark unhold linux-image-aws linux-headers-aws
```

## Understanding the Issue

- When you boot, Linux loads the kernel from `/boot`
- NVIDIA driver kernel modules must match the running kernel version
- These modules live in `/lib/modules/$(uname -r)/`
- DKMS (Dynamic Kernel Module Support) builds these modules
- DKMS requires kernel headers to build modules
- If headers are missing for new kernel → no NVIDIA modules → nvidia-smi fails

## System Info

Current setup:
- NVIDIA Driver: 580.105.08
- GPU: NVIDIA L4
- OS: Ubuntu 24.04 (noble)
- Kernel series: 6.14.0-aws (backported from newer Ubuntu)
