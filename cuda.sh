#!/bin/bash

set -e
set -o pipefail

# --- Terminal Colors ---
BOLD="\033[1m"
RED="\033[1;31m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
BLUE="\033[1;34m"
CYAN="\033[1;36m"
WHITE="\033[1;37m"
NORMAL="\033[0m"

# --- Logging Functions ---
log_header() { echo -e "\n${BOLD}${BLUE}=================================================${NORMAL}"; echo -e "${BOLD}${BLUE}# $1${NORMAL}"; echo -e "${BOLD}${BLUE}=================================================${NORMAL}"; }
log_step() { echo -e "\n${BOLD}${BLUE}>>> $1${NORMAL}"; }
log_info() { echo -e "${CYAN}    $1${NORMAL}"; }
log_success() { echo -e "${GREEN}    ✓ $1${NORMAL}"; }
log_warn() { echo -e "${YELLOW}    ! $1${NORMAL}"; }
log_error() { echo -e "${RED}${BOLD}!!! ERROR: $1${NORMAL}" >&2; }
log_detail() { echo -e "${WHITE}      - $1${NORMAL}"; }

# --- Cleanup Function ---
cleanup() {
    log_info "Running cleanup..."
    if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
    rm -f cuda-*.pin cuda-repo-*.deb cuda-keyring*.deb cuda_test cuda_test.cu
    log_info "Cleanup finished."
}
trap cleanup EXIT INT TERM

# --- Helper Functions ---
check_command() {
    command -v "$1" >/dev/null 2>&1
}

require_command() {
    log_info "Checking for command: $1"
    if ! check_command "$1"; then
        log_warn "Command '$1' not found. Attempting to install..."
        if check_command apt-get; then
            apt-get update >/dev/null || log_warn "apt-get update failed."
            apt-get install -y "$2" >/dev/null || {
                log_error "Failed to install '$2' which provides '$1'. Please install it manually and retry."
                return 1
            }
            log_success "Successfully installed '$2'."
        else
            log_error "Command '$1' not found, and 'apt-get' is not available to install '$2'. Please install '$1' manually."
            return 1
        fi
    fi
    log_success "Command '$1' found: $(command -v $1)"
    return 0
}

# --- Check Root Privileges ---
check_root() {
    log_step "Checking for Root Privileges"
    
    if [ "$(id -u)" -ne 0 ]; then
        log_error "This script must be run as root. Please use sudo."
        return 1
    fi
    log_success "Running as root."
    return 0
}

# --- System Information ---
gather_system_info() {
    log_header "System Information"
    
    # --- OS Detection ---
    if [ -f /etc/os-release ]; then
        source /etc/os-release
        OS_ID=$ID
        OS_VERSION_ID=$VERSION_ID
        OS_ID_LIKE=$ID_LIKE
        log_info "OS detected: $PRETTY_NAME"
    else
        log_warn "/etc/os-release not found. Attempting fallback detection."
        if check_command lsb_release; then
            OS_ID=$(lsb_release -is | tr '[:upper:]' '[:lower:]')
            OS_VERSION_ID=$(lsb_release -rs)
            log_info "OS detected using lsb_release: $OS_ID $OS_VERSION_ID"
        else
            log_error "Cannot determine operating system. This script requires /etc/os-release or lsb_release."
            return 1
        fi
    fi

    if [[ ! " $OS_ID $OS_ID_LIKE " =~ " debian " ]] && [[ ! " $OS_ID $OS_ID_LIKE " =~ " ubuntu " ]]; then
        log_error "This script is designed for Debian/Ubuntu-based systems. Detected OS: $OS_ID. Aborting."
        return 1
    fi
    log_success "Debian/Ubuntu based system confirmed: $OS_ID $OS_VERSION_ID"

    # --- Architecture Check ---
    ARCH=$(uname -m)
    log_info "Architecture: $ARCH"
    if [ "$ARCH" != "x86_64" ]; then
        log_error "This script currently only supports x86_64 architecture. Detected: $ARCH. Aborting."
        return 1
    fi
    log_success "Architecture x86_64 confirmed."

    # --- WSL Detection ---
    IS_WSL=false
    if grep -qi Microsoft /proc/version; then
        log_warn "WSL environment detected. Ensure you have installed the appropriate NVIDIA drivers on your Windows host."
        IS_WSL=true
    fi
    
    return 0
}

# --- GPU Detection ---
detect_gpu() {
    log_header "GPU Detection"
    
    if check_command nvidia-smi; then
        log_success "NVIDIA GPU detected via nvidia-smi."
        log_info "GPU Details:"
        nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv,noheader | while IFS=, read -r name driver pci;
        do
            log_detail "Name: $name, Driver Version: $driver, PCI ID: $pci"
        done
        
        # Extract driver version for later use
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d '[:space:]')
        log_success "NVIDIA Driver Version: $DRIVER_VERSION"
        
        # Extract CUDA compatibility version
        CUDA_COMPAT_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' || echo "unknown")
        if [ "$CUDA_COMPAT_VERSION" != "unknown" ]; then
            log_success "Driver CUDA Compatibility: $CUDA_COMPAT_VERSION"
            # Set this as our target CUDA version for installation
            CUDA_VERSION_TARGET=$CUDA_COMPAT_VERSION
            CUDA_VERSION_MAJOR_MINOR=$(echo "$CUDA_VERSION_TARGET" | sed 's/\./-/')
            log_success "Setting target CUDA version: $CUDA_VERSION_TARGET (Major-Minor: $CUDA_VERSION_MAJOR_MINOR)"
        else
            log_warn "Could not determine CUDA compatibility from driver."
            # Default if we can't detect
            CUDA_VERSION_TARGET="12.0"
            CUDA_VERSION_MAJOR_MINOR="12-0"
            log_warn "Using default CUDA version: $CUDA_VERSION_TARGET"
        fi
        
        return 0
    elif check_command lspci && lspci | grep -iq nvidia; then
        log_success "NVIDIA GPU detected via lspci (nvidia-smi not found)."
        log_info "GPU Details (from lspci):"
        lspci | grep -i nvidia | while IFS= read -r line; do log_detail "$line"; done
        log_warn "nvidia-smi command not found. Driver might be missing or not loaded."
        
        # Default if we can't detect from driver
        CUDA_VERSION_TARGET="12.0"
        CUDA_VERSION_MAJOR_MINOR="12-0"
        log_warn "Using default CUDA version: $CUDA_VERSION_TARGET"
        
        return 0
    else
        log_error "No NVIDIA GPU detected via nvidia-smi or lspci."
        log_error "An NVIDIA GPU is required for CUDA installation. Aborting."
        return 1
    fi
}

# --- Existing CUDA Check ---
check_existing_cuda() {
    log_header "Checking for Existing CUDA Installations"
    
    # Create arrays to store found CUDA installations
    declare -a CUDA_PATHS
    declare -a CUDA_VERSIONS
    FOUND_PATH_COUNT=0
    
    # Find installed CUDA via nvcc if it exists
    if check_command nvcc; then
        NVCC_PATH=$(command -v nvcc)
        NVCC_VERSION=$(nvcc --version | grep -oP 'release \K\d+\.\d+' || echo "unknown")
        CUDA_PATH=$(dirname "$(dirname "$NVCC_PATH")")
        
        log_success "Found CUDA Toolkit via nvcc:"
        log_detail "Path: $CUDA_PATH"
        log_detail "Version: $NVCC_VERSION"
        log_detail "nvcc location: $NVCC_PATH"
        
        # Add to found paths
        CUDA_PATHS[$FOUND_PATH_COUNT]=$CUDA_PATH
        CUDA_VERSIONS[$FOUND_PATH_COUNT]=$NVCC_VERSION
        FOUND_PATH_COUNT=$((FOUND_PATH_COUNT + 1))
    else
        log_warn "CUDA Toolkit (nvcc) not found in PATH."
    fi
    
    # Search common directories for CUDA installations
    log_info "Searching common directories for CUDA installations..."
    for dir_prefix in "/usr/local" "/opt" "/usr"; do
        while IFS= read -r -d $'\0' cuda_dir; do
            # Only process directories that look like CUDA installations
            if [ -d "$cuda_dir/bin" ] && [ -f "$cuda_dir/bin/nvcc" ]; then
                # Extract version from path if possible
                dir_version=$(echo "$cuda_dir" | grep -oP 'cuda-\K[0-9.]+' || echo "unknown")
                
                # Get version from nvcc if available
                if [ -x "$cuda_dir/bin/nvcc" ]; then
                    nvcc_version=$("$cuda_dir/bin/nvcc" --version 2>/dev/null | grep -oP 'release \K\d+\.\d+' || echo "$dir_version")
                else
                    nvcc_version=$dir_version
                fi
                
                log_success "Found CUDA installation:"
                log_detail "Path: $cuda_dir"
                log_detail "Version: $nvcc_version"
                
                # Add to found paths (if not already added from nvcc check)
                if [ "$cuda_dir" != "${CUDA_PATHS[0]}" ]; then
                    CUDA_PATHS[$FOUND_PATH_COUNT]=$cuda_dir
                    CUDA_VERSIONS[$FOUND_PATH_COUNT]=$nvcc_version
                    FOUND_PATH_COUNT=$((FOUND_PATH_COUNT + 1))
                fi
            fi
        done < <(find "$dir_prefix" -maxdepth 1 -name 'cuda*' -type d -print0 2>/dev/null)
    done
    
    # Report summary
    if [ $FOUND_PATH_COUNT -gt 0 ]; then
        log_info "Found $FOUND_PATH_COUNT CUDA installation(s):"
        for ((i=0; i<FOUND_PATH_COUNT; i++)); do
            log_detail "${CUDA_PATHS[$i]} (version: ${CUDA_VERSIONS[$i]})"
        done
        
        # Check if any match our target version
        MATCHING_INSTALL=""
        for ((i=0; i<FOUND_PATH_COUNT; i++)); do
            if [[ "${CUDA_VERSIONS[$i]}" == "$CUDA_VERSION_TARGET"* ]]; then
                MATCHING_INSTALL="${CUDA_PATHS[$i]}"
                log_success "Found a matching installation for target CUDA version $CUDA_VERSION_TARGET: $MATCHING_INSTALL"
                break
            fi
        done
        
        # Remember what we found for later
        CUDA_INSTALL_PATH=$MATCHING_INSTALL
        
        # Check for version mismatch with driver
        if [ -z "$MATCHING_INSTALL" ]; then
            log_warn "No CUDA installation matching driver compatibility version $CUDA_VERSION_TARGET was found."
            return 1  # No matching version found
        else
            return 0  # Matching version found
        fi
    else
        log_warn "No CUDA installations found."
        return 2  # No installation found
    fi
}

# --- Check for CUDA Runtime Libraries ---
check_cuda_runtime() {
    log_step "Checking for CUDA Runtime Libraries"
    
    # Check for libcudart
    if ldconfig -p | grep -q libcudart; then
        log_success "CUDA Runtime libraries found in system library path."
        return 0
    elif [ -n "$CUDA_INSTALL_PATH" ] && [ -d "$CUDA_INSTALL_PATH/lib64" ]; then
        if [ -f "$CUDA_INSTALL_PATH/lib64/libcudart.so" ]; then
            log_success "CUDA Runtime libraries found in CUDA installation directory."
            return 0
        fi
    fi
    
    log_warn "CUDA Runtime libraries not found or not properly linked."
    return 1
}

# --- Check CUDA PATH Setup ---
check_cuda_path() {
    log_step "Checking CUDA PATH Configuration"
    
    local missing_path=true
    
    # Check if CUDA bin is in PATH
    if [ -n "$CUDA_INSTALL_PATH" ]; then
        if echo "$PATH" | grep -q "$CUDA_INSTALL_PATH/bin"; then
            log_success "CUDA bin directory is in PATH: $CUDA_INSTALL_PATH/bin"
            missing_path=false
        else
            log_warn "CUDA bin directory is NOT in PATH: $CUDA_INSTALL_PATH/bin"
        fi
        
        # Check if CUDA lib64 is in LD_LIBRARY_PATH
        if echo "$LD_LIBRARY_PATH" | grep -q "$CUDA_INSTALL_PATH/lib64"; then
            log_success "CUDA lib64 directory is in LD_LIBRARY_PATH: $CUDA_INSTALL_PATH/lib64"
            missing_path=false
        else
            log_warn "CUDA lib64 directory is NOT in LD_LIBRARY_PATH: $CUDA_INSTALL_PATH/lib64"
        fi
    else
        log_warn "No valid CUDA installation path identified. Cannot check PATH configuration."
        return 1
    fi
    
    if $missing_path; then
        return 1
    fi
    return 0
}

# --- Setup PATH for this session ---
setup_current_session_path() {
    log_step "Setting up CUDA PATH for Current Session"
    
    if [ -z "$CUDA_INSTALL_PATH" ] || [ ! -d "$CUDA_INSTALL_PATH" ]; then
        log_error "No valid CUDA installation path available to set up PATH."
        return 1
    fi
    
    # Add CUDA bin to PATH if not already there
    if ! echo "$PATH" | grep -q "$CUDA_INSTALL_PATH/bin"; then
        export PATH="$CUDA_INSTALL_PATH/bin${PATH:+:${PATH}}"
        log_success "Added $CUDA_INSTALL_PATH/bin to PATH for current session."
    else
        log_info "$CUDA_INSTALL_PATH/bin already in PATH."
    fi
    
    # Add CUDA lib64 to LD_LIBRARY_PATH if not already there
    if ! echo "$LD_LIBRARY_PATH" | grep -q "$CUDA_INSTALL_PATH/lib64"; then
        export LD_LIBRARY_PATH="$CUDA_INSTALL_PATH/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        log_success "Added $CUDA_INSTALL_PATH/lib64 to LD_LIBRARY_PATH for current session."
    else
        log_info "$CUDA_INSTALL_PATH/lib64 already in LD_LIBRARY_PATH."
    fi
    
    # Verify nvcc is now in PATH
    if check_command nvcc; then
        log_success "nvcc is now available in PATH: $(command -v nvcc)"
        nvcc --version
        return 0
    else
        log_error "nvcc still not available in PATH after setting environment variables."
        return 1
    fi
}

# --- Configure Path for Future Sessions ---
setup_path_persistently() {
    log_step "Setting up CUDA PATH Persistently"
    
    if [ -z "$CUDA_INSTALL_PATH" ] || [ ! -d "$CUDA_INSTALL_PATH" ]; then
        log_error "No valid CUDA installation path available to set up persistent PATH."
        return 1
    fi
    
    # Create CUDA profile script
    log_info "Creating CUDA profile script for all users..."
    
    cat > /etc/profile.d/cuda.sh << EOL
#!/bin/sh
# CUDA Environment Variables - Added by CUDA setup script
export PATH="$CUDA_INSTALL_PATH/bin\${PATH:+:\${PATH}}"
export LD_LIBRARY_PATH="$CUDA_INSTALL_PATH/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
EOL
    
    chmod +x /etc/profile.d/cuda.sh
    log_success "Created /etc/profile.d/cuda.sh"
    
    # Update ldconfig
    log_info "Updating dynamic linker configuration..."
    
    if [ ! -f "/etc/ld.so.conf.d/cuda.conf" ]; then
        echo "$CUDA_INSTALL_PATH/lib64" > /etc/ld.so.conf.d/cuda.conf
        ldconfig
        log_success "Created /etc/ld.so.conf.d/cuda.conf and ran ldconfig"
    else
        log_info "File /etc/ld.so.conf.d/cuda.conf already exists. Checking content..."
        if ! grep -q "$CUDA_INSTALL_PATH/lib64" /etc/ld.so.conf.d/cuda.conf; then
            echo "$CUDA_INSTALL_PATH/lib64" >> /etc/ld.so.conf.d/cuda.conf
            ldconfig
            log_success "Updated /etc/ld.so.conf.d/cuda.conf and ran ldconfig"
        else
            log_info "CUDA lib64 path already in /etc/ld.so.conf.d/cuda.conf"
        fi
    fi
    
    # User-specific configuration
    if [ -n "$SUDO_USER" ]; then
        USER_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
        
        log_info "Configuring for user: $SUDO_USER (home: $USER_HOME)"
        
        # Check common profile files
        for profile_file in "$USER_HOME/.bashrc" "$USER_HOME/.zshrc" "$USER_HOME/.profile"; do
            if [ -f "$profile_file" ]; then
                if ! grep -q "CUDA.*PATH" "$profile_file"; then
                    log_info "Updating $profile_file..."
                    cat >> "$profile_file" << EOL

# CUDA Environment Variables - Added by CUDA setup script
export PATH="$CUDA_INSTALL_PATH/bin\${PATH:+:\${PATH}}"
export LD_LIBRARY_PATH="$CUDA_INSTALL_PATH/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
EOL
                    log_success "Updated $profile_file"
                else
                    log_info "CUDA PATH already configured in $profile_file"
                fi
            fi
        done
        
        # Set ownership
        chown "$SUDO_USER" "$USER_HOME/.bashrc" "$USER_HOME/.profile" 2>/dev/null || true
    else
        log_warn "No sudo user detected. Skipping user-specific profile configuration."
    fi
    
    log_success "CUDA PATH has been configured persistently. Changes will take effect after logging out and back in."
    log_info "To activate immediately for your user, run: source /etc/profile.d/cuda.sh"
    
    return 0
}

# --- Create Symbolic Links for Default CUDA Path ---
create_cuda_symlink() {
    log_step "Creating Default CUDA Symbolic Link"
    
    if [ -z "$CUDA_INSTALL_PATH" ] || [ ! -d "$CUDA_INSTALL_PATH" ]; then
        log_error "No valid CUDA installation path available to create symlink."
        return 1
    fi
    
    # Create /usr/local/cuda -> actual installation dir
    DEFAULT_CUDA_PATH="/usr/local/cuda"
    
    if [ -L "$DEFAULT_CUDA_PATH" ]; then
        log_info "Symbolic link $DEFAULT_CUDA_PATH already exists, pointing to: $(readlink -f $DEFAULT_CUDA_PATH)"
        if [ "$(readlink -f "$DEFAULT_CUDA_PATH")" = "$(readlink -f "$CUDA_INSTALL_PATH")" ]; then
            log_success "Symbolic link already points to the correct installation: $CUDA_INSTALL_PATH"
            return 0
        else
            log_warn "Symbolic link points to a different installation. Updating..."
            rm "$DEFAULT_CUDA_PATH"
        fi
    elif [ -e "$DEFAULT_CUDA_PATH" ]; then
        log_warn "$DEFAULT_CUDA_PATH exists but is not a symbolic link. Creating backup..."
        mv "$DEFAULT_CUDA_PATH" "${DEFAULT_CUDA_PATH}.bak.$(date +%s)"
    fi
    
    # Create the symlink
    ln -sf "$CUDA_INSTALL_PATH" "$DEFAULT_CUDA_PATH"
    log_success "Created symbolic link: $DEFAULT_CUDA_PATH -> $CUDA_INSTALL_PATH"
    
    return 0
}

# --- CUDA Installation ---
install_cuda() {
    log_header "Installing CUDA Toolkit $CUDA_VERSION_TARGET"
    
    # Determine OS-specific repository name
    if [ "$IS_WSL" = true ]; then
        OS_NAME_FOR_REPO="wsl-ubuntu"
    else
        case $OS_VERSION_ID in
            24.04*) OS_NAME_FOR_REPO="ubuntu2404" ;; 
            22.04*) OS_NAME_FOR_REPO="ubuntu2204" ;; 
            20.04*) OS_NAME_FOR_REPO="ubuntu2004" ;; 
            18.04*) OS_NAME_FOR_REPO="ubuntu1804" ;; 
            *) 
                log_warn "Unsupported Ubuntu/Debian version: $OS_VERSION_ID. Falling back to Ubuntu 22.04."
                OS_NAME_FOR_REPO="ubuntu2204" 
                ;; 
        esac
    fi
    log_info "Using OS configuration: $OS_NAME_FOR_REPO"
    
    # 1. Install CUDA keyring
    log_step "Installing CUDA repository keyring"
    
    KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${OS_NAME_FOR_REPO}/x86_64/cuda-keyring_1.1-1_all.deb"
    KEYRING_FILE="cuda-keyring.deb"
    
    log_info "Downloading keyring from $KEYRING_URL"
    if ! wget --quiet "$KEYRING_URL" -O "$KEYRING_FILE"; then
        log_warn "wget failed, trying curl..."
        if ! curl -s -L "$KEYRING_URL" -o "$KEYRING_FILE"; then
            log_error "Failed to download CUDA keyring. Please check your internet connection."
            return 1
        fi
    fi
    
    log_info "Installing keyring..."
    if ! dpkg -i "$KEYRING_FILE"; then
        log_error "Failed to install CUDA keyring."
        return 1
    fi
    rm -f "$KEYRING_FILE"
    log_success "CUDA keyring installed successfully."
    
    # 2. Update apt and install CUDA
    log_step "Updating package lists"
    apt-get update
    
    log_step "Installing CUDA Toolkit $CUDA_VERSION_TARGET"
    local cuda_package="cuda-toolkit-${CUDA_VERSION_MAJOR_MINOR}"
    
    log_info "Installing package: $cuda_package"
    log_warn "This may take several minutes. Please be patient."
    
    if ! apt-get install -y "$cuda_package"; then
        log_warn "Installation of specific package $cuda_package failed. Trying generic 'cuda' package..."
        if ! apt-get install -y cuda; then
            log_error "Failed to install CUDA Toolkit. Please check the error messages above."
            return 1
        fi
    fi
    
    # 3. Identify the installed CUDA path
    log_step "Identifying installed CUDA path"
    
    # First try to find the CUDA path based on the target version
    while IFS= read -r -d $'\0' cuda_dir; do
        if [[ "$cuda_dir" == *"cuda-$CUDA_VERSION_TARGET"* ]] && [ -d "$cuda_dir/bin" ] && [ -f "$cuda_dir/bin/nvcc" ]; then
            CUDA_INSTALL_PATH="$cuda_dir"
            log_success "Found installed CUDA path matching target version: $CUDA_INSTALL_PATH"
            break
        fi
    done < <(find /usr/local /opt /usr -maxdepth 1 -name "cuda*${CUDA_VERSION_TARGET}*" -type d -print0 2>/dev/null)
    
    # If not found, try to find any CUDA installation
    if [ -z "$CUDA_INSTALL_PATH" ]; then
        while IFS= read -r -d $'\0' cuda_dir; do
            if [ -d "$cuda_dir/bin" ] && [ -f "$cuda_dir/bin/nvcc" ]; then
                CUDA_INSTALL_PATH="$cuda_dir"
                log_success "Found installed CUDA path: $CUDA_INSTALL_PATH"
                break
            fi
        done < <(find /usr/local /opt /usr -maxdepth 1 -name "cuda*" -type d -print0 2>/dev/null)
    fi
    
    # Fall back to default location if still not found
    if [ -z "$CUDA_INSTALL_PATH" ]; then
        CUDA_INSTALL_PATH="/usr/local/cuda-$CUDA_VERSION_TARGET"
        log_warn "Could not find installed CUDA path. Assuming default: $CUDA_INSTALL_PATH"
    fi
    
    log_success "CUDA installation completed successfully!"
    return 0
}

# --- Verify CUDA Installation ---
verify_cuda() {
    log_header "Verifying CUDA Installation"
    
    # Check if nvcc is available
    if ! check_command nvcc; then
        log_error "CUDA compiler (nvcc) not found in PATH after installation."
        return 1
    fi
    
    # Get nvcc version
    NVCC_VERSION=$(nvcc --version | grep -oP 'release \K\d+\.\d+' || echo "unknown")
    log_success "CUDA compiler (nvcc) found: $(command -v nvcc)"
    log_detail "NVCC Version: $NVCC_VERSION"
    
    # Create temp directory for test compilation
    TEMP_DIR=$(mktemp -d)
    log_info "Using temporary directory for verification: $TEMP_DIR"
    
    # Change to temp directory
    cd "$TEMP_DIR"
    
    # Create simple CUDA test program
    log_step "Creating CUDA test program"
    cat > cuda_test.cu << 'EOL'
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGPU() {
    printf("Hello from GPU thread %d, block %d\n", threadIdx.x, blockIdx.x);
}

int main() {
    // CPU Hello
    printf("Hello from CPU!\n");
    
    // Get device count
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    // Report device info
    printf("Found %d CUDA devices\n", deviceCount);
    
    // Launch kernel
    helloFromGPU<<<1, 1>>>();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    // Wait for GPU to finish
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        printf("cudaDeviceSynchronize error: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    printf("CUDA test completed successfully!\n");
    return 0;
}
EOL
    
    # Compile CUDA test program
    log_step "Compiling CUDA test program"
    if ! nvcc cuda_test.cu -o cuda_test; then
        log_error "Failed to compile CUDA test program. Check nvcc output above."
        return 1
    fi
    log_success "CUDA test program compiled successfully."
    
    # Run CUDA test program
    log_step "Running CUDA test program"
    ./cuda_test
    
    if [ $? -eq 0 ]; then
        log_success "CUDA test program ran successfully!"
        return 0
    else
        log_error "CUDA test program failed. See output above for details."
        return 1
    fi
}

# --- Print System Info ---
print_system_info() {
    log_header "CUDA System Information"
    
    if check_command nvidia-smi; then
        log_step "NVIDIA System Management Interface (nvidia-smi)"
        nvidia-smi
    else
        log_warn "nvidia-smi not found in PATH."
    fi
    
    if check_command nvcc; then
        log_step "CUDA Compiler Information (nvcc)"
        nvcc --version
    else
        log_warn "nvcc not found in PATH."
    fi
    
    log_step "Environment Variables"
    log_detail "PATH: $PATH"
    log_detail "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
    
    if [ -n "$CUDA_INSTALL_PATH" ]; then
        log_detail "CUDA Installation Path: $CUDA_INSTALL_PATH"
    fi
    
    log_step "CUDA Installations"
    find /usr/local /opt /usr -maxdepth 1 -name "cuda*" -type d 2>/dev/null | while read -r dir; do
        if [ -d "$dir/bin" ] && [ -f "$dir/bin/nvcc" ]; then
            local version
            version=$("$dir/bin/nvcc" --version 2>/dev/null | grep -oP 'release \K[0-9.]+' || echo "unknown")
            log_detail "$dir (CUDA version: $version)"
        fi
    done
    
    # Check for CUDA libraries
    log_step "CUDA Libraries"
    ldconfig -p | grep -i cuda | head -n 10 | while read -r line; do
        log_detail "$line"
    done
    
    log_success "System information gathering complete."
}

# --- Print Usage Information ---
print_usage() {
    cat << EOL
CUDA Setup Script
Usage: sudo bash $0 [options]

This script detects, installs, and configures NVIDIA CUDA on Debian/Ubuntu systems.

Options:
  --help          Show this help message and exit
  --install       Force installation even if CUDA is already detected
  --skip-verify   Skip the verification step after installation
  --info-only     Only display system information, don't install or configure anything

Example:
  sudo bash $0 --install     # Force installation of CUDA
EOL
}

# --- Print Final Instructions ---
print_final_instructions() {
    log_header "Next Steps"
    
    cat << EOL
${BOLD}${GREEN}
CUDA setup completed successfully!
${NORMAL}

${BOLD}To use CUDA in the current terminal session:${NORMAL}
    source /etc/profile.d/cuda.sh

${BOLD}To verify CUDA is working:${NORMAL}
    nvcc --version
    nvidia-smi

${BOLD}The changes will take full effect after logging out and back in.${NORMAL}

${BOLD}Important paths:${NORMAL}
  • CUDA Installation: $CUDA_INSTALL_PATH
  • CUDA Default Link: /usr/local/cuda

${BOLD}If you encounter any issues:${NORMAL}
  1. Check that the NVIDIA driver is properly installed
  2. Ensure the CUDA paths are in your PATH and LD_LIBRARY_PATH
  3. Verify your GPU is compatible with this CUDA version

${BOLD}For development, you may want to install these additional packages:${NORMAL}
  • NVIDIA cuDNN: High-performance neural network library
    sudo apt install libcudnn8 libcudnn8-dev

  • NVIDIA TensorRT: High-performance deep learning inference optimizer
    sudo apt install libnvinfer8 libnvinfer-dev

${BOLD}For Python development:${NORMAL}
  • Install CUDA support for frameworks like PyTorch or TensorFlow
    pip install torch torchvision torchaudio

${YELLOW}Note: When installing ML frameworks, ensure they're compatible
with your installed CUDA version: $NVCC_VERSION${NORMAL}
EOL
}

# --- Run All Steps ---
main() {
    # Parse command line arguments
    local force_install=false
    local skip_verify=false
    local info_only=false
    
    for arg in "$@"; do
        case $arg in
            --help)
                print_usage
                exit 0
                ;;
            --install)
                force_install=true
                ;;
            --skip-verify)
                skip_verify=true
                ;;
            --info-only)
                info_only=true
                ;;
            *)
                echo "Unknown option: $arg"
                print_usage
                exit 1
                ;;
        esac
    done
    
    # Quick info only mode
    if $info_only; then
        if ! check_root; then
            log_error "Root privileges required for system information."
            exit 1
        fi
        gather_system_info
        detect_gpu
        check_existing_cuda
        print_system_info
        exit 0
    fi
    
    # Regular execution
    if ! check_root; then
        log_error "This script must be run as root. Please use sudo."
        exit 1
    fi
    
    # Start workflow
    gather_system_info || exit 1
    detect_gpu || exit 1
    
    # Check if we need to install
    local need_install=true
    if check_existing_cuda; then
        if $force_install; then
            log_warn "CUDA already installed, but --install flag provided. Proceeding with installation."
        else
            log_success "Compatible CUDA installation found. Skipping installation."
            need_install=false
        fi
    else
        log_info "No compatible CUDA installation found. Proceeding with installation."
    fi
    
    # Install CUDA if needed
    if $need_install; then
        # Install required packages for CUDA installation
        log_step "Installing prerequisites"
        apt-get update
        apt-get install -y build-essential wget curl software-properties-common
        
        # Install CUDA
        install_cuda || exit 1
    fi
    
    # Set up environment
    setup_current_session_path || log_warn "Failed to set up CUDA path for current session."
    setup_path_persistently || log_warn "Failed to set up CUDA path persistently."
    create_cuda_symlink || log_warn "Failed to create CUDA symbolic link."
    
    # Verify installation
    if ! $skip_verify; then
        verify_cuda || log_warn "CUDA verification failed. Installation may be incomplete or incompatible."
    else
        log_info "Skipping verification as requested."
    fi
    
    # Print final system info
    print_system_info
    print_final_instructions
    
    log_header "CUDA Setup Complete"
    log_success "CUDA has been successfully installed and configured!"
    
    return 0
}

# --- Execute Main Function ---
main "$@"
