from time import time
from de.samplers.maskers import ImportanceMasker, RandomMasker
from de.common.utils import split_kmers


WT_SEQ = "PVMHPHGAPPNHRPWQMKDLQAIKQEVSQAAPGSPQFMQTIRLAVQQFDPTAKDLQDLLQYLCSSLVASLHHQQLDSLISEAETRGITGYNPLAGPLRVQANNPQQQGLRREYQQLWLAAFAALP"


def test_random_masker():
    wt_seq = WT_SEQ
    population = [wt_seq] * 8
    population = split_kmers(population, k=3)

    masker = RandomMasker(max_subs=3)
    st = time()
    masked_pop, masked_pos = masker.run(population)
    et = time()
    print(f"Time to run masker.run: {et - st}")
    for pos, pop in zip(masked_pos, masked_pop):
        print(pos)
        print(pop)
        print('=============\n')


def test_importance_masker():
    wt_seq = WT_SEQ
    masker = ImportanceMasker(
        use_full_dataset=False,
        full_dataset=None,
        max_subs=3,
        low_importance_mask=True
    )
    population = [wt_seq] * 8
    population = split_kmers(population, k=3)
    st = time()
    masked_pop, masked_pos = masker.run(population)
    et = time()
    print(f"Time to run masker.run: {et - st}")
    for pos, pop in zip(masked_pos, masked_pop):
        print(pos)
        print(pop)
        print('=============\n')


# test_random_masker()
test_importance_masker()
