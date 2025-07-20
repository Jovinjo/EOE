ALL_ID_DATASET = [
    'bird200', 'pet37', 'cub100_ID', 'pet18_ID', 'cifar10_ID'
]

ALL_OOD_TASK = [
    'far', 'fine_grained', 'near'
]

ALL_LLM = [
    'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4'
]

dataset_mappings = {
    # far OOD mappings
    'bird200': ['dtd'],
    'pet37': ['dtd'],
    
    # fine-grained OOD mappings
    'cub100_ID': ['cub100_OOD'],
    'pet18_ID': ['pet19_OOD'],
    
    # near OOD mappings
    'cifar10_ID': ['cifar100_OOD']
}