import subprocess

def install_packages_from_requirements(file_path='requirements.txt'):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                pkg = line.strip()
                
                # Skip empty lines and comments
                if not pkg or pkg.startswith('#'):
                    continue

                print(f"Installing: {pkg}")
                result = subprocess.run(['pip', 'install', pkg])

                if result.returncode != 0:
                    print(f"❌ Failed to install: {pkg}")
                else:
                    print(f"✅ Successfully installed: {pkg}")

    except FileNotFoundError:
        print(f"Error: '{file_path}' not found.")

if __name__ == "__main__":
    install_packages_from_requirements()
