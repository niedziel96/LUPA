import os 

def get_nested_path(root_path, tissue_tag, magnification):
    nested_path = os.path.join(root_path, tissue_tag, str(magnification))

    return nested_path
