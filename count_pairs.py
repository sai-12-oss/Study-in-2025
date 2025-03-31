import ctypes
def count_pairs_file(filename: str) -> int:
    # Load the shared library
    lib = ctypes.CDLL('./final.so')

    # Define the function prototype
    lib.solve.restype = ctypes.c_int
    lib.solve.argtypes = [ctypes.c_char_p]

    # Call the C function
    input_str = filename
    return lib.solve(input_str.encode('utf-8'))

