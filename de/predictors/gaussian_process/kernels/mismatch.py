"""
    Python implementation of mismatch string kernel
    Based on: `https://github.com/jakob-he/string-kernel/`
"""

import numpy as np


class MismatchTrie(object):
    """
    Trie implementation, specific to 'Mismatch String Kernels'.
    """

    def __init__(self, label=None, parent=None):
        """
        label: int, optional (default None), node label.
        parent: `Trie` instance, optional (default None), node's parent.
        """

        self.label = label  # label on edge connecting this node to its parent
        self.level = 0  # level of this node beyond the root node
        self.children = {}  # children of this node

        # concatenation of all labels of nodes from root node to this node
        self.full_label = ""
        # for each sample string, this dict holds pointers to it's k-mer substrings
        self.kmers = {}

        self.parent = parent

        if parent is not None:
            parent.add_child(self)

    def is_root(self):
        """
        Check whether this node is the root.
        """

        return self.parent is None

    def is_leaf(self):
        """
        Check whether this node is a leaf.
        """

        return len(self.children) == 0

    def is_empty(self):
        """
        Check whether a node has 'died'.
        """

        return len(self.kmers) == 0

    def copy_kmers(self):
        """
        Copy the kmer data for this node (not the reference pointer).
        """

        return {
            index: np.array(substring_pointers)
            for index, substring_pointers in self.kmers.items()
        }

    def add_child(self, child):
        """
        Add a new child to this node.
        """
        assert child.label not in self.children

        # initialize kmers data to that of parent
        child.kmers = self.copy_kmers()

        # child is one level beyond parent
        child.level = self.level + 1

        # parent's full label (concatenation of labels on edges leading from root node)
        # is a prefix to child's the remainder is one symbol, the child's label
        child.full_label = '%s%s' % (self.full_label, child.label)

        # let parent adopt child: commit child to parent's booklist
        self.children[child.label] = child

        # let child adopt parent
        child.parent = self

    def delete_child(self, child):
        """
        Delete a child.
        """

        # get child label
        label = child.label if isinstance(child, MismatchTrie) else child

        # check that child really exists
        assert label in self.children, "No child with label %s exists." % label

        # delete the child
        del self.children[label]

    def compute_kmers(self, training_data, k):
        """
        Compute the metadata for this node, i.e, for each input string
        training_data[index], compute the list of offsets of it's k-mers
        together with the mismatch counts (intialially zero) for this k-mers
        with the k-mer represented by this node `self`.

        Parameters
        ----------
        training_data: 2D array of shape (n_samples, n_features)
                       training data for the kernel.
        k: int, used in k-mers to compute the kernel.
        """

        # sanity checks
        if not isinstance(training_data, np.ndarray):
            training_data = np.array(training_data)

        if training_data.ndim == 1:
            training_data = np.array([training_data])

        assert training_data.ndim == 2

        # compute the len(training_data[index]) - k + 1 kmers of each input training string
        for index in range(len(training_data)):
            self.kmers[index] = np.array([
                (
                    offset,
                    0  # no mismatch yet
                ) for offset in range(len(training_data[index]) - k + 1)
            ])

    def process_node(self, training_data, k, m):
        """
        Process this node. Recompute its supported k-mers.
        Finally, determine if node survives.

        Parameters
        ----------
        training_data: 2D array of shape (n_samples, n_features)
                       training data for the kernel
        k: int, used in k-mers to compute the kernel
        m: int
           maximum number of mismatches for 2 k-mers to be considered 'similar'
           Normally small values of m should work well
           Plus, the complexity the algorithm is exponential in m

        Return
        -------
        True if node survives, False else
        """

        # sanity checks
        if not isinstance(training_data, np.ndarray):
            training_data = np.array(training_data)
        if training_data.ndim == 1:
            training_data = np.array([training_data])

        assert training_data.ndim == 2

        if self.is_root():
            # compute meta-data
            self.compute_kmers(training_data, k)
        else:
            # loop on all k-mers of input string training_data[index]
            for index, substring_pointers in self.kmers.items():
                # update mismatch counts
                substring_pointers[..., 1] += (
                    training_data[index][substring_pointers[..., 0] +
                                         self.level - 1] != self.label)

                # delete substring_pointers that present more than m mismatches
                self.kmers[index] = np.delete(
                    substring_pointers,
                    np.nonzero(substring_pointers[..., 1] > m),
                    axis=0)

            # delete entries with empty substring_pointer list
            self.kmers = {
                index: substring_pointers
                for (index, substring_pointers) in self.kmers.items()
                if len(substring_pointers)
            }

        return not self.is_empty()

    def update_kernel(self, kernel):
        """
        Update the kernel in memory.

        Parameters
        ----------
        kernel: 2D array of shape (n_samples, n_samples)
                kernel to be updated
        full_label: mismatch kmers generated by traversing labels from root to leaf
        """

        for i in self.kmers:
            for j in self.kmers:
                kernel[i, j] += len(self.kmers[i]) * len(self.kmers[j])

    def traverse(self,
                 training_data,
                 l,
                 k,
                 m,
                 kernel=None,
                 kernel_update_callback=None):
        """
        Traverses a node, expanding it to plausible descendants.

        Parameters
        ----------
        training_data: 2D array of shape (n_samples, n_features)
                       training data for the kernel
        l: int, size of alphabet
           Examples of values with a natural interpretation:
           2: for binary data
           256: for data encoded as strings of bytes
           4: for DNA/RNA sequence data (bioinformatics)
           20: for protein data (bioinformatics)
        k: int, we will use k-mers to compute the kernel
        m: int
           maximum number of mismatches for 2 k-mers to be considered 'similar'
           Normally small values of m should work well
           Plus, the complexity the algorithm is exponential in m
        kernel: 2D array of shape (n_samples, n_samples)
                optional (default None) kernel to be, or being, estimated

        Returns
        -------
        kernel: 2D array of shape (n_samples, n_samples), estimated kernel
        n_survived_kmers: int, number of leaf nodes that survived the traversal
        go_ahead: boolean, a flag indicating whether the node got aborted (False) or not
        """

        # initialize kernel if None
        if kernel is None:
            kernel = np.zeros((len(training_data), len(training_data)))

        # counts the number of leafs which are decendants of this node
        n_surviving_kmers = 0

        # process the node
        go_ahead = self.process_node(training_data, k, m)

        # if the node survived
        if go_ahead:
            # we've hit a leaf
            if k == 0:
                # yes, this is one more leaf/kmer
                n_surviving_kmers += 1

                # update the kernel
                self.update_kernel(kernel)
            else:
                # recursively bear and traverse child nodes
                for j in range(l):
                    # bear child
                    child = MismatchTrie(label=j, parent=self)

                    # traverse child
                    kernel, child_n_surviving_kmers, child_go_ahead = child.traverse(
                        training_data, l, k - 1, m, kernel=kernel)

                    # delete child if dead
                    if child.is_empty():
                        self.delete_child(child)

                    # update leaf counts
                    n_surviving_kmers += child_n_surviving_kmers if \
                        child_go_ahead else 0

        return kernel, n_surviving_kmers, go_ahead

    def __iter__(self):
        """
        Return an iterator on the nodes of the trie
        """

        yield self
        for child in self.children.values():
            for grandchild in child:
                yield grandchild

    def leafs(self):
        for leaf in self:
            if leaf.is_leaf():
                yield leaf


def integerized(sequence):
    """
    Convert the character string into numeric string.
    """

    key_dict = sorted(set(sequence))
    int_seq = []
    for char in sequence:
        to_int = key_dict.index(char)
        int_seq.append(to_int)

    return int_seq


def preprocess(sequences):
    """
    Data preprocessing for string sequences.
    """

    upper_seq = []
    len_record = []
    for seq in sequences:
        seq = seq.upper()
        upper_seq.append(integerized(seq))
        len_record.append(len(seq))

    length_used = min(len_record)
    post_seq = [seq[:length_used] for seq in upper_seq]

    return post_seq


def normalize_kernel(kernel):
    """
    Normalizes a kernel[x, y] by doing:
    kernel[x, y] / sqrt(kernel[x, x] * kernel[y, y])
    """

    nkernel = np.copy(kernel)

    assert nkernel.ndim == 2
    assert nkernel.shape[0] == nkernel.shape[1]

    denom = np.einsum("ii,jj->ij", nkernel, nkernel)
    denom = np.sqrt(denom, out=denom)
    nkernel = np.divide(nkernel, denom, out=nkernel)

    # Set diagonal elements as 1
    np.fill_diagonal(nkernel, 1.)

    return nkernel


class MismatchKernel(MismatchTrie):
    """
    Python implementation of Mismatch String Kernels.
    Parameters
    ----------
    l: int, optional (default None), size of alphabet.
       Examples of values with a natural interpretation:
       2: for binary data
       256: for data encoded as strings of bytes
       4: for DNA/RNA sequence (bioinformatics)
       20: for protein data (bioinformatics)
    k: int, optional (default None), the k in 'k-mer'.
    m: int, optional (default None)
       maximum number of mismatches for 2 k-mers to be considered 'similar'.
       Normally small values of m should work well.
       Plus, the complexity of the algorithm is exponential in m.
    **kwargs: dict, optional (default empty)
              optional parameters to pass to `tree.MismatchTrie` instantiation.

    Attributes
    ----------
    `kernel`: 2D array of shape (n_sampled, n_samples), estimated kernel.
    `n_survived_kmers`: number of leafs/k-mers that survived trie traversal.
    """

    def __init__(self, l: int = None, k: int = None, m: int = None, **kwargs):

        if None not in [l, k, m]:

            # invoke trie.MismatchTrie constructor
            MismatchTrie.__init__(self, **kwargs)

            # sanitize alphabet size
            if l < 2:
                raise ValueError(
                    "Alphabet too small. l must be at least 2; got %i" % l)

            # sanitize kernel parameters (k, m)
            if 2 * m > k:
                raise ValueError(
                    ("You provided k = %i and m = %i. m is too big (must"
                     "be at ""must k / 2). This doesn't make sense.") % (k, m))

            self.l = l
            self.k = k
            self.m = m

    def get_kernel(self, X, normalize=True, **kwargs):
        """
        Main calling function to get mismatch string kernel.
        """

        if isinstance(X, tuple):
            assert len(X) == 5, "Invalid model."
            self.l, self.k, self.m, self.leaf_kmers, self.kernel = X
            # sanitize the types and shapes of self.l, self.j, self.m,
            # self.leaf_kmers, and self.kernel
        else:
            # traverse/build trie proper
            for x in ['l', 'k', 'm']:
                if not hasattr(self, x):
                    raise RuntimeError(
                        ("'%s' not specified during object initialization."
                         "You must now specify complete model (tuple of l, "
                         "k, m, leafs, and, kernel).") % x)
            self.kernel, _, _ = self.traverse(
                X, self.l, self.k, self.m, **kwargs)

            # normalize kernel
            if normalize:
                self.kernel = normalize_kernel(self.kernel)

            # gather up the leafs
            self.leaf_kmers = dict((leaf.full_label,
                                    dict((index, len(kgs))
                                         for index, kgs in leaf.kmers.items()))
                                    for leaf in self.leafs())

        return self
