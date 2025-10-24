import os
from typing import List, Union, Iterator, Optional
from urbaning.data.sequence import Sequence  # assuming Sequence is defined in this module


class Dataset:
    """
    Represents a collection of sequences within a dataset directory.

    This class manages access to multiple sequences, allowing iteration,
    indexing, and automatic loading from a root folder.

    Attributes
    ----------
    root_folder : str
        Path to the root directory of the dataset.
    sequences : list of str
        List of sequence names or identifiers contained in the dataset.

    Examples
    --------
    >>> dataset = Dataset("/data/urbaning")
    >>> len(dataset)
    5
    >>> sequence = dataset[0]
    >>> for seq in dataset:
    ...     print(seq.name)
    """

    def __init__(self, root_folder: str, sequences: Union[List[str], str, None] = None):
        """
        Initialize a dataset from a root folder and optional sequence list.

        Parameters
        ----------
        root_folder : str
            Path to the dataset's root directory.
        sequences : list of str, str, or None, optional
            - If a list: treated as explicit sequence names.
            - If a string: path to a text file containing one sequence name per line.
            - If None: automatically loads all sequences in the dataset root folder `root_folder/dataset`.
        """
        if sequences is None:
            sequences = os.listdir(os.path.join(root_folder, 'dataset'))
        elif isinstance(sequences, str):
            with open(sequences, "r") as f:
                sequences = [line.strip() for line in f.readlines()]

        self.root_folder: str = root_folder
        self.sequences: List[str] = sequences

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)

    def __getitem__(self, item: int) -> Sequence:
        """
        Retrieve a `Sequence` object at the specified index.

        Parameters
        ----------
        item : int
            Index of the sequence to retrieve.

        Returns
        -------
        Sequence
            Sequence object corresponding to the selected index.
        """
        return Sequence(self.root_folder, self.sequences[item])

    def __iter__(self) -> Iterator[Sequence]:
        """
        Iterate over all sequences in the dataset.

        Yields
        ------
        Sequence
            Sequence object for each entry in the dataset.
        """
        for i in range(len(self)):
            yield self[i]
