#!/bin/bash

set -e
set -o pipefail

# --- Configuration ---
CUDA_VERSION_TARGET="12.8"
CUDA_VERSION_MAJOR_MINOR="12-8"

# --- Stylish Logging ---
# Check if tput is available for more advanced styling
if command -v tput >/dev/null 2>&1; then
    BOLD=$(tput bold)
    UNDERLINE=$(tput smul)
    NORMAL=$(tput sgr0)
    RED=$(tput setaf 1)
    GREEN=$(tput setaf 2)
    YELLOW=$(tput setaf 3)
    BLUE=$(tput setaf 4)
    MAGENTA=$(tput setaf 5)
    CYAN=$(tput setaf 6)
    WHITE=$(tput setaf 7)
else
    BOLD="\033[1m"
    UNDERLINE="\033[4m"
    NORMAL="\033[0m"
    RED="\033[1;31m"
    GREEN="\033[1;32m"
    YELLOW="\033[1;33m"
    BLUE="\033[1;34m"
    MAGENTA="\033[1;35m"
    CYAN="\033[1;36m"
    WHITE="\033[1;37m"
fi

log_step() { echo -e "\n${BLUE}${BOLD}>>> $1${NORMAL}"; }
log_info() { echo -e "${CYAN}    $1${NORMAL}"; }
log_success() { echo -e "${GREEN}    $1${NORMAL}"; }
log_warn() { echo -e "${YELLOW}    $1${NORMAL}"; }
log_error() { echo -e "${RED}${BOLD}!!! ERROR: $1${NORMAL}"; } >&2
log_detail() { echo -e "${WHITE}      - $1${NORMAL}"; }

# --- Cleanup Function ---
cleanup() {
    log_info "Running cleanup..."
    rm -f cuda-*.pin cuda-repo-*.deb cuda-keyring*.deb cuda_test cuda_test.cu
    if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
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
                exit 1
            }
            log_success "Successfully installed '$2'."
            if ! check_command "$1"; then
                 log_error "Installation of '$2' succeeded but command '$1' is still not found. Please check your PATH or installation."
                 exit 1
            fi
        else
            log_error "Command '$1' not found, and 'apt-get' is not available to install '$2'. Please install '$1' manually."
            exit 1
        fi
    else
        log_success "Command '$1' found: $(command -v $1)"
    fi
}

# --- Pre-flight Checks ---
log_step "Performing Pre-flight Checks"

if [ "$(id -u)" -ne 0 ]; then
   log_error "This script must be run as root. Please use sudo."
   exit 1
fi
log_success "Running as root."

require_command wget wget
require_command curl curl
require_command dpkg dpkg
require_command apt-get apt-utils
require_command lsb_release lsb-release
require_command lspci pciutils

# --- System Information ---
log_step "Gathering System Information"

OS_ID=""
OS_VERSION_ID=""
OS_ID_LIKE=""
if [ -f /etc/os-release ]; then
    source /etc/os-release
    OS_ID=$ID
    OS_VERSION_ID=$VERSION_ID
    OS_ID_LIKE=$ID_LIKE
    log_info "OS detected from /etc/os-release: $PRETTY_NAME"
else
    log_warn "/etc/os-release not found. Attempting fallback detection."
    if check_command lsb_release; then
        OS_ID=$(lsb_release -is | tr '[:upper:]' '[:lower:]')
        OS_VERSION_ID=$(lsb_release -rs)
        log_info "OS detected using lsb_release: $OS_ID $OS_VERSION_ID"
    else
        log_error "Cannot determine operating system. This script requires /etc/os-release or lsb_release."
        exit 1
    fi
fi

if [[ ! " $OS_ID $OS_ID_LIKE " =~ " debian " ]] && [[ ! " $OS_ID $OS_ID_LIKE " =~ " ubuntu " ]]; then
    log_error "This script is designed for Debian/Ubuntu-based systems. Detected OS: $OS_ID. Aborting."
    exit 1
fi
log_success "Debian/Ubuntu based system confirmed."

ARCH=$(uname -m)
log_info "Architecture: $ARCH"
if [ "$ARCH" != "x86_64" ]; then
    log_error "This script currently only supports x86_64 architecture. Detected: $ARCH. Aborting."
    exit 1
fi
log_success "Architecture x86_64 confirmed."

IS_WSL=false
if grep -qi Microsoft /proc/version; then
    log_warn "WSL environment detected. Ensure you have installed the appropriate NVIDIA drivers on your Windows host."
    IS_WSL=true
fi

# --- GPU Detection ---
check_nvidia_gpu() {
    log_step "Checking for NVIDIA GPU"
    if check_command nvidia-smi; then
        log_success "NVIDIA GPU detected via nvidia-smi."
        log_info "GPU Details:"
        nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv,noheader | while IFS=, read -r name driver pci;
        do
            log_detail "Name: $name, Driver: $driver, PCI ID: $pci"
        done
        return 0
    elif check_command lspci && lspci | grep -iq nvidia; then
        log_success "NVIDIA GPU detected via lspci (nvidia-smi not found)."
        log_info "GPU Details (from lspci):"
        lspci | grep -i nvidia | while IFS= read -r line; do log_detail "$line"; done
        log_warn "nvidia-smi command not found. Driver might be missing or not loaded. CUDA installation might fail."
        return 0
    else
        log_error "No NVIDIA GPU detected via nvidia-smi or lspci."
        log_error "An NVIDIA GPU is required for CUDA installation. Aborting."
        return 1
    fi
}

# --- CUDA Installation Check ---
check_cuda_installed() {
    log_step "Checking for Existing CUDA Installation"

    local cuda_toolkit_found=false
    local cuda_driver_found=false
    local nvcc_path=""
    local detected_cuda_version=""
    local detected_driver_cuda_version=""

    if check_command nvcc; then
        cuda_toolkit_found=true
        nvcc_path=$(command -v nvcc)
        detected_cuda_version=$(nvcc --version | grep -oP 'release \K\d+\.\d+' || echo "unknown")
        log_success "CUDA Toolkit (nvcc) found: Version $detected_cuda_version at $nvcc_path"
    else
        log_warn "CUDA Toolkit (nvcc) not found in PATH."
    fi

    if check_command nvidia-smi; then
        cuda_driver_found=true
        detected_driver_cuda_version=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' || echo "unknown")
        log_success "NVIDIA driver found with CUDA compatibility: $detected_driver_cuda_version"
    else
        log_warn "NVIDIA driver (nvidia-smi) not found or not loaded."
    fi

    local found_paths=()
    for path_prefix in /usr/local /opt /usr; do
        while IFS= read -r -d $'\0' cuda_dir; do
            if [ -f "$cuda_dir/bin/nvcc" ]; then
                found_paths+=("$cuda_dir")
                log_success "Found potential CUDA installation at: $cuda_dir"
                if ! $cuda_toolkit_found; then
                    cuda_toolkit_found=true # Found via path even if not in PATH
                fi
            fi
        done < <(find "$path_prefix" -maxdepth 1 -name 'cuda*' -type d -print0 2>/dev/null)
    done

    if $cuda_toolkit_found && $cuda_driver_found; then
        log_success "Existing CUDA Toolkit and compatible driver detected."
        if [[ "$detected_cuda_version" == "$CUDA_VERSION_TARGET"* ]]; then
            log_success "Detected version $detected_cuda_version matches target $CUDA_VERSION_TARGET. Installation likely not needed."
            return 0 # Found matching version
        elif [[ "$detected_cuda_version" != "unknown" ]]; then
            log_warn "Detected CUDA version $detected_cuda_version differs from target $CUDA_VERSION_TARGET."
            return 1 # Found different version
        else
             log_warn "Could not determine exact version of installed CUDA Toolkit."
             return 1 # Found but version unknown
        fi
    elif $cuda_toolkit_found; then
        log_warn "CUDA Toolkit found, but NVIDIA driver (nvidia-smi) was not found or is not loaded."
        return 1 # Toolkit only
    elif $cuda_driver_found; then
        log_warn "NVIDIA driver found, but CUDA Toolkit (nvcc) was not found."
        return 1 # Driver only
    else
        log_info "No existing CUDA Toolkit or NVIDIA driver detected."
        return 2 # Nothing found
    fi
}

# --- Installation Functions ---
install_cuda_local_deb() {
    log_step "Starting CUDA Installation (Method: Local Deb)"

    local PIN_FILE=""
    local PIN_URL=""
    local DEB_FILE=""
    local DEB_URL=""
    local OS_NAME_FOR_REPO=""
    local OS_VERSION_FOR_REPO=""

    if [ "$IS_WSL" = true ]; then
        log_info "Configuring for WSL installation..."
        OS_NAME_FOR_REPO="wsl-ubuntu"
        PIN_FILE="cuda-${OS_NAME_FOR_REPO}.pin"
        PIN_URL="https://developer.download.nvidia.com/compute/cuda/repos/${OS_NAME_FOR_REPO}/x86_64/${PIN_FILE}"
        DEB_FILE="cuda-repo-${OS_NAME_FOR_REPO}-${CUDA_VERSION_MAJOR_MINOR}-local_${CUDA_VERSION_TARGET}.0-1_amd64.deb"
        DEB_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION_TARGET}.0/local_installers/${DEB_FILE}"
    else
        case $OS_VERSION_ID in
            24.04*)
                OS_NAME_FOR_REPO="ubuntu2404"
                OS_VERSION_FOR_REPO="2404"
                ;;
            22.04*)
                OS_NAME_FOR_REPO="ubuntu2204"
                OS_VERSION_FOR_REPO="2204"
                ;;
            20.04*)
                OS_NAME_FOR_REPO="ubuntu2004"
                OS_VERSION_FOR_REPO="2004"
                ;;
            18.04*)
                OS_NAME_FOR_REPO="ubuntu1804"
                OS_VERSION_FOR_REPO="1804"
                ;;
            *)
                log_warn "Unsupported Ubuntu/Debian version: $OS_VERSION_ID. Falling back to Ubuntu 22.04 configuration."
                OS_NAME_FOR_REPO="ubuntu2204"
                OS_VERSION_FOR_REPO="2204"
                ;;
        esac
        log_info "Configuring for $OS_NAME_FOR_REPO..."
        PIN_FILE="cuda-${OS_NAME_FOR_REPO}.pin"
        PIN_URL="https://developer.download.nvidia.com/compute/cuda/repos/${OS_NAME_FOR_REPO}/x86_64/${PIN_FILE}"
        # Note: The exact deb filename might vary slightly based on driver version bundled
        # We will try a pattern or rely on the alternative method if this specific name fails.
        # Constructing a likely name based on common patterns:
        DEB_FILE_PATTERN="cuda-repo-${OS_NAME_FOR_REPO}-${CUDA_VERSION_MAJOR_MINOR}-local_${CUDA_VERSION_TARGET}*.deb"
        DEB_URL_BASE="https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION_TARGET}.0/local_installers/"
        # Attempt to find the exact DEB URL - this is fragile
        # A better approach might be to use the network repo method first
        DEB_FILE="cuda-repo-${OS_NAME_FOR_REPO}-${CUDA_VERSION_MAJOR_MINOR}-local_${CUDA_VERSION_TARGET}.0-1_amd64.deb" # Default guess
        if [[ "$OS_VERSION_FOR_REPO" == "2404" ]]; then
             # Example for 24.04, might need adjustment based on actual NVIDIA releases
             DEB_FILE="cuda-repo-ubuntu2404-${CUDA_VERSION_MAJOR_MINOR}-local_${CUDA_VERSION_TARGET}.0-*.deb" # Use wildcard
             # Need a way to fetch the exact name or use network repo
             log_warn "Local deb filename for Ubuntu 24.04 is uncertain. Download might fail. Consider network repo method."
             # Fallback to a known older version's naming scheme as a guess
             DEB_FILE="cuda-repo-ubuntu2404-${CUDA_VERSION_MAJOR_MINOR}-local_${CUDA_VERSION_TARGET}.0-555.42.02-1_amd64.deb" # Example, likely wrong
        fi
        DEB_URL="${DEB_URL_BASE}${DEB_FILE}"

    fi

    log_info "Downloading repository pin file: $PIN_FILE"
    log_detail "URL: $PIN_URL"
    rm -f "$PIN_FILE"
    if ! wget --quiet "$PIN_URL" -O "$PIN_FILE"; then
        log_warn "wget failed for pin file, trying curl..."
        if ! curl -fsSL "$PIN_URL" -o "$PIN_FILE"; then
            log_error "Failed to download pin file: $PIN_FILE from $PIN_URL"
            return 1
        fi
    fi
    log_success "Downloaded $PIN_FILE."

    log_info "Setting up repository preferences..."
    mkdir -p /etc/apt/preferences.d
    if ! cp "$PIN_FILE" /etc/apt/preferences.d/cuda-repository-pin-600; then
        log_error "Failed to copy $PIN_FILE to /etc/apt/preferences.d/"
        rm -f "$PIN_FILE"
        return 1
    fi
    log_success "Repository preferences set."

    log_info "Downloading CUDA local repository package: $DEB_FILE"
    log_detail "URL: $DEB_URL"
    rm -f "$DEB_FILE"
    if ! wget --progress=bar:force "$DEB_URL" -O "$DEB_FILE"; then
        log_warn "wget failed for deb file, trying curl..."
        if ! curl -L --progress-bar "$DEB_URL" -o "$DEB_FILE"; then
             log_error "Failed to download CUDA local repository package: $DEB_FILE from $DEB_URL"
             log_error "This might be due to an incorrect filename guess or network issues."
             rm -f "$PIN_FILE" "$DEB_FILE"
             return 1
        fi
    fi

    if [ ! -f "$DEB_FILE" ] || [ ! -s "$DEB_FILE" ]; then
        log_error "Downloaded file $DEB_FILE is missing or empty."
        rm -f "$PIN_FILE" "$DEB_FILE"
        return 1
    fi
    log_success "Downloaded $DEB_FILE."

    log_info "Installing CUDA local repository package..."
    if ! dpkg -i "$DEB_FILE"; then
        log_error "Failed to install $DEB_FILE. Attempting dependency fix..."
        if ! apt-get --fix-broken install -y; then
             log_error "'apt-get --fix-broken install' failed. Cannot proceed with local deb method."
             rm -f "$PIN_FILE" "$DEB_FILE"
             return 1
        fi
        # Retry dpkg install after fixing dependencies
        if ! dpkg -i "$DEB_FILE"; then
            log_error "Failed to install $DEB_FILE even after fixing dependencies."
            rm -f "$PIN_FILE" "$DEB_FILE"
            return 1
        fi
    fi
    log_success "Installed $DEB_FILE."

    log_info "Copying repository keys..."
    local key_copied=false
    for key_path in /var/cuda-repo-*/cuda-*-keyring.gpg; do
        if [ -f "$key_path" ]; then
            if ! cp "$key_path" /usr/share/keyrings/; then
                log_error "Failed to copy CUDA keyring from $key_path"
                rm -f "$PIN_FILE" "$DEB_FILE"
                return 1
            fi
            log_success "Copied keyring from $key_path"
            key_copied=true
            break # Assume first one found is correct
        fi
    done
    if ! $key_copied; then
        log_error "Could not find CUDA keyring GPG file in /var/cuda-repo-*/."
        rm -f "$PIN_FILE" "$DEB_FILE"
        return 1
    fi

    log_info "Updating package list..."
    if ! apt-get update; then
        log_error "apt-get update failed after adding local CUDA repo."
        rm -f "$PIN_FILE" "$DEB_FILE"
        return 1
    fi
    log_success "Package list updated."

    log_info "Installing CUDA Toolkit $CUDA_VERSION_TARGET..."
    log_warn "This may take several minutes. Please be patient."
    local cuda_package="cuda-toolkit-${CUDA_VERSION_MAJOR_MINOR}"
    if ! apt-get install -y "$cuda_package"; then
        log_warn "Installation of $cuda_package failed. Trying generic 'cuda' package..."
        if ! apt-get install -y cuda; then
            log_error "Failed to install CUDA Toolkit using both specific and generic package names."
            rm -f "$PIN_FILE" "$DEB_FILE"
            return 1
        fi
        log_success "Installed CUDA using generic 'cuda' package."
    else
        log_success "Installed $cuda_package successfully."
    fi

    log_success "CUDA Toolkit installed successfully via Local Deb method!"
    rm -f "$PIN_FILE" "$DEB_FILE"
    return 0
}

install_cuda_network_repo() {
    log_step "Starting CUDA Installation (Method: Network Repo)"

    local OS_NAME_FOR_REPO=""
    local REPO_URL=""
    local KEYRING_URL=""
    local KEYRING_FILE="cuda-keyring.deb"

     if [ "$IS_WSL" = true ]; then
        log_info "Configuring for WSL installation..."
        OS_NAME_FOR_REPO="wsl-ubuntu"
    else
        case $OS_VERSION_ID in
            24.04*) OS_NAME_FOR_REPO="ubuntu2404" ;; 
            22.04*) OS_NAME_FOR_REPO="ubuntu2204" ;; 
            20.04*) OS_NAME_FOR_REPO="ubuntu2004" ;; 
            18.04*) OS_NAME_FOR_REPO="ubuntu1804" ;; 
            *) log_warn "Unsupported Ubuntu/Debian version: $OS_VERSION_ID. Falling back to Ubuntu 22.04."; OS_NAME_FOR_REPO="ubuntu2204" ;; 
        esac
    fi
    log_info "Using configuration for: $OS_NAME_FOR_REPO"

    REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/${OS_NAME_FOR_REPO}/x86_64/"
    KEYRING_URL="${REPO_URL}cuda-keyring_1.1-1_all.deb"

    log_info "Downloading CUDA apt keyring..."
    log_detail "URL: $KEYRING_URL"
    rm -f "$KEYRING_FILE"
    if ! wget --progress=bar:force "$KEYRING_URL" -O "$KEYRING_FILE"; then
        log_warn "wget failed for keyring, trying curl..."
        if ! curl -L --progress-bar "$KEYRING_URL" -o "$KEYRING_FILE"; then
            log_error "Failed to download keyring: $KEYRING_FILE from $KEYRING_URL"
            rm -f "$KEYRING_FILE"
            return 1
        fi
    fi

    if [ ! -f "$KEYRING_FILE" ] || [ ! -s "$KEYRING_FILE" ]; then
        log_error "Downloaded file $KEYRING_FILE is missing or empty."
        rm -f "$KEYRING_FILE"
        return 1
    fi
    log_success "Downloaded $KEYRING_FILE."

    log_info "Installing CUDA apt keyring..."
    if ! dpkg -i "$KEYRING_FILE"; then
        log_error "Failed to install $KEYRING_FILE. Attempting dependency fix..."
         if ! apt-get --fix-broken install -y; then
             log_error "'apt-get --fix-broken install' failed. Cannot proceed with network repo method."
             rm -f "$KEYRING_FILE"
             return 1
         fi
         if ! dpkg -i "$KEYRING_FILE"; then
             log_error "Failed to install $KEYRING_FILE even after fixing dependencies."
             rm -f "$KEYRING_FILE"
             return 1
         fi
    fi
    log_success "Installed $KEYRING_FILE."

    log_info "Updating package list..."
    if ! apt-get update; then
        log_error "apt-get update failed after adding network CUDA repo."
        rm -f "$KEYRING_FILE"
        return 1
    fi
    log_success "Package list updated."

    log_info "Installing CUDA Toolkit $CUDA_VERSION_TARGET..."
    log_warn "This may take several minutes. Please be patient."
    local cuda_package="cuda-toolkit-${CUDA_VERSION_MAJOR_MINOR}"
    if ! apt-get install -y "$cuda_package"; then
        log_warn "Installation of $cuda_package failed. Trying generic 'cuda' package..."
        if ! apt-get install -y cuda; then
            log_error "Failed to install CUDA Toolkit using both specific and generic package names."
            rm -f "$KEYRING_FILE"
            return 1
        fi
        log_success "Installed CUDA using generic 'cuda' package."
    else
        log_success "Installed $cuda_package successfully."
    fi

    log_success "CUDA Toolkit installed successfully via Network Repo method!"
    rm -f "$KEYRING_FILE"
    return 0
}

# --- Post-Installation Setup & Verification ---
setup_cuda_env_instructions() {
    log_step "Post-Installation: Environment Setup Instructions"

    local cuda_install_path=""
    if check_command nvcc; then
        local nvcc_loc
        nvcc_loc=$(command -v nvcc)
        cuda_install_path=$(dirname "$(dirname "$nvcc_loc")")
        log_info "Detected CUDA installation path: $cuda_install_path"
    else
        log_warn "nvcc not found in PATH after installation. Searching common locations..."
        local potential_path=""
        for path_prefix in /usr/local /opt /usr; do
            while IFS= read -r -d $'\0' cuda_dir; do
                if [ -f "$cuda_dir/bin/nvcc" ]; then
                    potential_path="$cuda_dir"
                    break 2 # Found one, exit both loops
                fi
            done < <(find "$path_prefix" -maxdepth 1 -name "cuda*${CUDA_VERSION_MAJOR_MINOR}*" -type d -print0 2>/dev/null)
        done
        if [ -n "$potential_path" ]; then
             cuda_install_path=$potential_path
             log_info "Found potential CUDA path: $cuda_install_path"
        else
             cuda_install_path="/usr/local/cuda-${CUDA_VERSION_TARGET}" # Fallback guess
             log_warn "Could not automatically determine CUDA installation path. Assuming default: $cuda_install_path"
        fi
    fi

    if [ -z "$cuda_install_path" ] || [ ! -d "$cuda_install_path" ]; then
        log_error "Failed to determine a valid CUDA installation directory. Cannot provide environment setup instructions."
        return 1
    fi

    log_info "To make CUDA available in your shell sessions, add the following lines to your shell profile (~/.bashrc, ~/.zshrc, ~/.profile, etc.):"
    echo -e "${BOLD}"
    echo -e "export PATH=\"${cuda_install_path}/bin\"]\${PATH:+:\${PATH}}"
    echo -e "export LD_LIBRARY_PATH=\"${cuda_install_path}/lib64\"]\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
    echo -e "${NORMAL}"
    log_info "After adding these lines, reload your shell configuration (e.g., 'source ~/.bashrc') or log out and log back in."
    log_warn "This script does NOT automatically modify your user profile files."

    # Set for current session for verification
    export PATH="${cuda_install_path}/bin"${PATH:+:${PATH}}
    export LD_LIBRARY_PATH="${cuda_install_path}/lib64"${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    log_info "Environment variables set for the current script execution."
    return 0
}

verify_cuda() {
    log_step "Verifying CUDA Installation"

    if ! check_command nvcc; then
        log_error "nvcc command not found in PATH. Verification failed."
        log_error "Please ensure CUDA bin directory is in your PATH (see previous step)."
        return 1
    fi
    log_success "CUDA compiler (nvcc) found: $(command -v nvcc)"
    log_info "NVCC Version:"
    nvcc --version

    if ! check_command nvidia-smi; then
        log_warn "nvidia-smi command not found. Cannot verify driver interaction."
    else
        log_success "NVIDIA driver (nvidia-smi) found: $(command -v nvidia-smi)"
        log_info "NVIDIA SMI Output:"
        nvidia-smi
    fi

    log_info "Attempting to compile and run a sample CUDA program..."
    TEMP_DIR=$(mktemp -d)
    log_info "Using temporary directory: $TEMP_DIR"
    pushd "$TEMP_DIR" > /dev/null

    cat > cuda_test.cu << 'EOL'
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGPU() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Hello World from GPU! Thread %d, Block %d\n", threadIdx.x, blockIdx.x);
    }
}

int main() {
    printf("Hello World from CPU!\n");
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        printf("Could not enumerate CUDA devices. Is the driver loaded and compatible?\n");
        return 1;
    }
    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 1;
    }
    printf("Found %d CUDA device(s)\n", deviceCount);
    
    helloFromGPU<<<1, 1>>>();
    err = cudaGetLastError(); // Check for kernel launch errors
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaDeviceSynchronize(); // Wait for kernel to complete
    if (err != cudaSuccess) {
        printf("cudaDeviceSynchronize error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("CUDA Test Program Completed Successfully.\n");
    return 0;
}
EOL
    log_info "Created sample file: cuda_test.cu"

    log_info "Compiling with nvcc..."
    if ! nvcc cuda_test.cu -o cuda_test; then
        log_error "nvcc compilation failed."
        popd > /dev/null
        rm -rf "$TEMP_DIR"
        return 1
    fi
    log_success "Compilation successful: cuda_test executable created."

    log_info "Running the compiled test program..."
    if ! ./cuda_test; then
        log_error "CUDA test program execution failed."
        log_error "Check the output above for CUDA errors. Ensure drivers are loaded and compatible."
        popd > /dev/null
        rm -rf "$TEMP_DIR"
        return 1
    fi

    log_success "CUDA test program executed successfully!"
    popd > /dev/null
    rm -rf "$TEMP_DIR"
    log_success "CUDA Verification Complete!"
    return 0
}

# --- Main Execution Logic ---
main() {
    log_step "Starting CUDA Installation Script"

    if ! check_nvidia_gpu; then
        exit 1
    fi

    cuda_status=2 # 0=matching version, 1=different/unknown version or driver/toolkit mismatch, 2=nothing found
    check_cuda_installed
    cuda_status=$?

    if [ $cuda_status -eq 0 ]; then
        log_success "Matching CUDA version ($CUDA_VERSION_TARGET) already installed. Verification recommended."
        read -p "Do you want to proceed with verification anyway? [y/N]: " -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            if ! setup_cuda_env_instructions; then exit 1; fi
            if ! verify_cuda; then exit 1; fi
            log_success "Verification successful."
        else
            log_info "Skipping verification. Exiting."
            exit 0
        fi
    else
        if [ $cuda_status -eq 1 ]; then
            log_warn "Existing CUDA installation or driver found, but it doesn't match the target version ($CUDA_VERSION_TARGET) or is incomplete."
            read -p "Do you want to attempt installation of CUDA $CUDA_VERSION_TARGET anyway? This might overwrite or conflict with existing components. [y/N]: " -r response
            if [[ ! "$response" =~ ^[Yy]$ ]]; then
                log_info "Aborting installation as requested."
                exit 0
            fi
        fi

        log_info "Proceeding with CUDA $CUDA_VERSION_TARGET installation."
        # Try Network Repo method first as it's generally more reliable
        if install_cuda_network_repo; then
            log_success "CUDA installation via Network Repo successful."
        else
            log_warn "Network Repo installation method failed. Attempting Local Deb method..."
            if install_cuda_local_deb; then
                log_success "CUDA installation via Local Deb successful."
            else
                log_error "Both Network Repo and Local Deb installation methods failed."
                log_error "Please check the logs above for specific errors."
                log_error "Common issues include network problems, incorrect OS/version detection, or incompatible hardware/drivers."
                exit 1
            fi
        fi

        # Post-install steps
        if ! setup_cuda_env_instructions; then exit 1; fi
        if ! verify_cuda; then exit 1; fi
    fi

    log_step "CUDA Installation and Verification Complete!"
    log_success "Remember to configure your shell environment as instructed above."
}

# --- Run Main Function ---
main

exit 0
