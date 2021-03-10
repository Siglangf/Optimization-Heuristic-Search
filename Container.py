class Container:
    def __init__(self, id, weight):
        self.id = id
        self.weight = weight

    def __repr__(self):
        return f"ID: {self.id}, WEIGHT:{self.weight}"


def sortArrayWeightDescending(Container_array):
    return sorted(Container_array, key=lambda x: x.weight, reverse=True)


def sortArrayWeightAscending(Container_array):
    return sorted(Container_array, key=lambda x: x.weight, reverse=False)
