# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# print(torch.cuda.is_available())
# print(torch.backends.cudnn.is_available())
# print(torch.cuda_version)
# print(torch.backends.cudnn.version())

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    import torch
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
