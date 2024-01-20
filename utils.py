def write_to_file(filename, data):
    with open(filename, 'w') as file:
        file.write(data)

def read_from_file(filename):
    with open(filename, 'r') as file:
        return file.read()

def delete_file(filename):
    import os
    if os.path.exists(filename):
        os.remove(filename)
        print(f"The file '{filename}' has been deleted.")
    else:
        print(f"The file '{filename}' does not exist.")
