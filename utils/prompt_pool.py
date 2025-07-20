class PromptGenerator:
    def __init__(self):
        self.dataset_info = {
            'bird200': {'class_type': None, 'num_classes': 200},
            'pet37': {'class_type': None, 'num_classes': 37},
            'cub100_ID': {'class_type': 'bird', 'num_classes': 100},
            'dtd47': {'class_type': 'texture', 'num_classes': 47},
            'pet18_ID': {'class_type': 'pet (including dogs and cats)', 'num_classes': 18},
            'cifar10_ID': {'class_type': None, 'num_classes': 10},
        }

    def _fine_grained_prompt(self, class_type, num_classes, class_info, envision_nums=50):
        return f"""Q: I have a dataset containing 10 unique species of dogs. I need a list of 10 distinct dog species that are NOT present in my dataset, and ensure there are no repetitions in the list you provide. For context, the species in my dataset are: ['husky dog', 'alaskan Malamute', 'cossack sled dog', 'golden retriever', 'German Shepherd', 'Beagle', 'Bulldog', 'Poodle', 'Dachshund', 'Doberman Pinscher']
A: The other 10 dog species not in the dataset are:
- Labrador Retriever
- Rottweiler
- Boxer
- Border Collie
- Shih Tzu
- Akita
- Saint Bernard
- Australian Shepherd
- Great Dane
- Boston Terrier

Q: I have a dataset containing {num_classes} different species of {class_type}. I need a list of {envision_nums} distinct {class_type} species that are NOT present in my dataset, and ensure there are no repetitions in the list you provide. For context, the species in my dataset are: {class_info}
A: The other {envision_nums} {class_type} species not in the dataset are:
"""

    def _far_prompt(self, class_type, num_classes, class_info, envision_nums=50):
        return f"""Q: I have gathered images of 4 distinct categories: ['Husky dog', 'Garfield cat', 'churches', 'truck']. Summarize what broad categories these categories might fall into based on visual features. Now, I am looking to identify 5 categories that visually resemble these broad categories but have no direct relation to them. Please list these 5 items for me.
A: These 5 items are:
- black stone
- mountain
- Ginkgo Tree
- river
- Rapeseed

Q: I have gathered images of {num_classes} distinct categories: [{class_info}]. Summarize what broad categories these categories might fall into based on visual features. Now, I am looking to identify {envision_nums} classes that visually resemble these broad categories but have no direct relation to them. Please list these {envision_nums} items for me.
A: These {envision_nums} items are:
"""

    def _near_prompt(self, class_type, num_classes, class_info, envision_nums=3):
        return f"""Q: Given the image category [husky dog], please suggest visually similar categories that are not directly related or belong to the same primary group as [husky dog]. Provide suggestions that share visual characteristics but are from broader and different domains than [husky dog].
A: There are 3 classes similar to [husky dog], and they are from broader and different domains than [husky dog]:
- gray wolf
- black stone
- red panda

Q: Given the image category [{class_info}], please suggest visually similar categories that are not directly related or belong to the same primary group as [{class_info}]. Provide suggestions that share visual characteristics but are from broader and different domains than [{class_info}].
A: There are {envision_nums} classes similar to [{class_info}], and they are from broader and different domains than [{class_info}]:
"""

    def _fine_grained_prompt_again(self, in_dataset, envision_nums=50):
        class_type = self.dataset_info[in_dataset]["class_type"]
        return f"""Q: Provide {envision_nums} additional {class_type} categories that aren't in the set I gave you, and haven't been mentioned in your previous responses to me.
A: The {envision_nums} additional categories are:
"""

    def _far_prompt_again(self, in_dataset, envision_nums=50):
        return f"""Q: Give me {envision_nums} more categories that are visually similar to these broad categories you summarized in the dataset but have no direct relation to them. Each category you give cannot exceed three words and should not have appeared in your previous answers.
A: The other {envision_nums} categories are:
"""

    def get_prompt(self, ood_task, in_dataset, class_info=None, envision_nums=50):
        dispatcher = {
            'fine_grained': self._fine_grained_prompt,
            'far': self._far_prompt,
            'near': self._near_prompt
        }
        if ood_task not in dispatcher:
            raise ValueError(f"Unknown OOD task: {ood_task}")
        info = self.dataset_info[in_dataset]
        return dispatcher[ood_task](info["class_type"], info["num_classes"], class_info, envision_nums)

    def get_prompt_again(self, ood_task, in_dataset, class_info=None, envision_nums=50):
        dispatcher = {
            'fine_grained': self._fine_grained_prompt_again,
            'far': self._far_prompt_again,
        }
        if ood_task not in dispatcher:
            raise ValueError(f"Unknown OOD task for repeat prompting: {ood_task}")
        return dispatcher[ood_task](in_dataset, envision_nums)
