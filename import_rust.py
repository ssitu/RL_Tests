import os
import shutil


def copy_compiled_rust(lib_name: str, release: bool = True):
    """
    Helper function to copy the compiled rust library to the current directory
    :param lib_name: The name of the library to copy
    :return: None
    """
    target_folder = "release" if release else "debug"
    try:
        src = f"{lib_name}/target/{target_folder}/{lib_name}.dll"
        dst = f"{lib_name}.pyd"
        # Copy the compiled rust library to the current directory and change the extension to .pyd
        if (not os.path.exists(dst)) or (os.stat(src).st_mtime - os.stat(dst).st_mtime > 1):
            shutil.copy2(src, dst)
            print(f"Updated {lib_name}.pyd")
    except FileNotFoundError as e:
        print(f"Error copying {lib_name} to current directory: {e}")


if __name__ == "__main__":
    copy_compiled_rust("rust_utils")
