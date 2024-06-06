



def no_transform(args):
    return args


def transform_images(transform_x, transform_y=no_transform):
    def _(args):
        print(args)
        return transform_x(args[0]), transform_y(args[1])
    return _


def transform_celebA(transform_x):
    def _(args):
        print(args)
        return transform_x(args["image"])
    return _


class TransformedDataset:
    def __init__(self, dataset, transforms=no_transform):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.transforms(self.dataset[item])
