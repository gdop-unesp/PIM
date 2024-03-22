import pathlib
import os
import variablesPIM  


def createDirIfDoesntExist(baseDirectory, directory: str) -> None:
    """
    Creates a new directory at the specified path if it doesn't already exist.

    Args:
        baseDirectory (str): The base directory path to create the new directory in.
        directory (str): The name of the directory to create.

    Returns:
        None

    Example usage:
        >>> createDirIfDoesntExist("C:\\my_folder", "new_dir")

    Note:
        This function checks if a directory exists at the specified path before attempting to create a new directory.
        If a directory with the specified name already exists, the function does nothing.
    """
    if not pathlib.Path(os.path.join(baseDirectory,directory)).is_dir():
        print(f'Creating directory {directory}')
        os.mkdir(str(os.path.join(baseDirectory,directory)))
   
def check_existence(directory_path: str, extension: str) -> list:
    """
    Returns a list of file paths with the specified extension in a given directory and its subdirectories.

    Args:
        directory_path (str): The directory path to search for files.
        extension (str): The file extension to search for (e.g. ".txt").

    Returns:
        list: A list of file paths that match the specified extension.

    Raises:
        SystemExit: If there are no files with the specified extension in the directory.

    Example usage:
        >>> files = check_existence("my_folder", ".txt")
        >>> print(files)
        ['my_folder\\file1.txt', 'my_folder\\subdir\\file2.txt']

    Note:
        This function searches for files with the specified extension in the given directory and its subdirectories.
        If there are no files with the specified extension in the directory, the function will print an error message and exit the program.
    """
    file_list = []
    print(directory_path)
    files = os.listdir(directory_path)
    files = [file for file in files if file.lower().endswith(extension.lower())]
    print(f"Arquivos {extension} encontrados: {files}")

    for root, dirs, files in os.walk(directory_path):
        print(files)
        for file in files:
            if extension.upper() in file.upper():
                file_list.append(os.path.join(root, file))
    if not file_list:
        raise FileNotFoundError(f"No files found with '{extension}' extension.")
    else:
        return file_list


def check_directory_existence(directory_path):
    """
    Verifies the existence of a directory.

    Args:
        directory_path (str): The path of the directory to be checked.

    Returns:
        bool: True if the directory exists, False otherwise.
    """
    return os.path.exists(directory_path)



def createFileWrite(fileDirectory, nome):
    """
    Opens a file in write mode and returns a file object.

    Args:
        fileDirectory (str): The directory path where the file should be created.
        nome (str): The name of the file to be created.

    Returns:
        file: A file object that can be used to write to the file.

    Raises:
        OSError: If the file cannot be created or opened.

    Example usage:
        >>> file_obj = createFileWrite("my_folder", "my_file.txt")
        >>> file_obj.write("This is the content of my file.")
        >>> file_obj.close()

    Note:
        This function assumes that the file should be created within the project's working directory. It uses the `variablesPIM` module to construct the full file path.
        If a file with the same name already exists in the specified directory, it will be overwritten by the new file.
    """
    try:
        file_path = str(variablesPIM.directory.joinpath(fileDirectory).joinpath(nome).resolve())
        return open(file_path, 'w+')
    except OSError:
        raise OSError("Error: Can't create or open the file.")

def createFileRead(fileDirectory, nome):
    """
    Opens a file in read mode and returns a file object.

    Args:
        fileDirectory (str): The directory path where the file is located.
        nome (str): The name of the file to be opened.

    Returns:
        file: A file object that can be used to read the file.

    Raises:
        OSError: If the file cannot be opened or read.

    Example usage:
        >>> file_obj = createFileRead("my_folder", "my_file.txt")
        >>> contents = file_obj.read()
        >>> print(contents)
        'This is the content of my file.'

    Note:
        This function assumes that the file is located within the project's working directory. It uses the `variablesPIM` module to construct the full file path.
    """
    try:
        file_path = str(variablesPIM.directory.joinpath(fileDirectory).joinpath(nome).resolve())
        return open(file_path, 'r')
    except OSError:
        raise OSError("Error: Can't open and read the file.")

def fileToList(file):
    """
    Reads a file and returns its content as a list of strings.

    Args:
        file (str): The path to the file to read.

    Returns:
        list: A list of strings, where each string represents a line in the file.

    Raises:
        OSError: If the file cannot be opened or read.

    Example usage:
        >>> content = fileToList("my_file.txt")
        >>> print(content)
        ['line 1', 'line 2', 'line 3']
    """
    try:
        with open(file) as f:
            content = f.readlines()
        #content = [x.rstrip("\n") for x in content]
        return content
    except OSError as e:
        print("Error: Can't open and read the file.")
        return None


def hasExtension(fileList, extension):
    for fileName in fileList:
        if fileName.endswith(extension):
            return True
    return False